#!/usr/bin/env python3
"""
Agente de Nutrição/Treinos — CrewAI + Redis (histórico) + JSON (dados) + WhatsApp
---------------------------------------------------------------------------------
Objetivo: demo fácil pra mostrar pelo WhatsApp (WABot x-api-key), com logs claros,
Redis guardando **histórico de interação** e JSON como "banco de dados" de perfil/dados.
✔ Webhook aceita o payload no formato que você enviou (array com headers/body/data)
✔ Envia resposta via API do WABot (x-api-key): [https://wabot.ggailabs.com/whatsapp/message/sendText](https://wabot.ggailabs.com/whatsapp/message/sendText)
✔ Endpoints REST também existem pra testar sem WhatsApp
✔ Conversação natural e humana com IA
✔ Usa modelo direto da variável de ambiente MODEL
✔ Pronto pra EasyPanel (ver Dockerfile e requirements no final deste arquivo)

Instalação local:
  pip install fastapi uvicorn httpx python-dotenv redis "crewai>=0.60.0" pydantic<3
  export OPENROUTER_API_KEY=sk-or-v1-...
  export MODEL=z-ai/glm-4.5-air:free
  export REDIS_URL=redis://localhost:6379/0
  export WABOT_API_KEY=SEU_TOKEN
  uvicorn wellness_agent_whatsapp_json_redis:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import httpx
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process

# -------------------------
# Setup & Config
# -------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("wellness")

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BRAND = os.getenv("BRAND", "Fitly.ai")

# -------------------------
# OpenRouter Configuration - CORRIGIDO PARA USAR DIRETAMENTE A VARIÁVEL MODEL
# -------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    log.error("❌ OPENROUTER_API_KEY não configurada!")
    raise ValueError("OPENROUTER_API_KEY é obrigatória")

# Configurar variáveis que o LiteLLM espera para OpenRouter
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_API_BASE"] = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")

# USAR DIRETAMENTE A VARIÁVEL MODEL (SEM MAPEAMENTO)
MODEL = os.getenv("MODEL", "z-ai/glm-4.5-air:free")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

log.info(f"✅ Modelo configurado: {MODEL}")

# WhatsApp sender (WABot) - HTTPS
WABOT_BASE = os.getenv("WABOT_BASE", "https://wabot.ggailabs.com")
WABOT_SEND_PATH = "/whatsapp/message/sendText"
WABOT_API_KEY = os.getenv("WABOT_API_KEY", "changeme")

# Redis cliente
try:
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    log.info(f"✅ Redis conectado: {REDIS_URL}")
except Exception as e:
    log.error(f"❌ Erro ao conectar Redis: {e}")
    raise

# -------------------------
# Modelos Webhook
# -------------------------
class EvoMessageData(BaseModel):
    jid: str
    sender: str
    isGroup: bool
    pushName: Optional[str] = None
    type: str
    text: Optional[str] = None
    media: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class EvoBody(BaseModel):
    event: str
    timestamp: Optional[str] = None
    data: EvoMessageData

class EvoEnvelope(BaseModel):
    headers: Dict[str, Any]
    params: Dict[str, Any]
    query: Dict[str, Any]
    body: EvoBody
    webhookUrl: Optional[str] = None
    executionMode: Optional[str] = None

def envelope_from_any(item: Dict[str, Any]) -> EvoEnvelope:
    """Converte payloads em formatos flexíveis para EvoEnvelope"""
    if isinstance(item, dict) and item.get("body") and isinstance(item["body"], dict):
        return EvoEnvelope.model_validate({
            "headers": item.get("headers", {}),
            "params": item.get("params", {}),
            "query": item.get("query", {}),
            "body": EvoBody.model_validate(item["body"]),
            "webhookUrl": item.get("webhookUrl"),
            "executionMode": item.get("executionMode"),
        })
    
    if isinstance(item, dict) and "event" in item and "data" in item:
        return EvoEnvelope.model_validate({
            "headers": item.get("headers", {}),
            "params": item.get("params", {}),
            "query": item.get("query", {}),
            "body": EvoBody.model_validate(item),
        })
    
    if isinstance(item, dict) and "data" in item and isinstance(item["data"], dict):
        body = {
            "event": item.get("event", "message.received"), 
            "timestamp": item.get("timestamp"), 
            "data": item["data"]
        }
        return EvoEnvelope.model_validate({
            "headers": item.get("headers", {}),
            "params": item.get("params", {}),
            "query": item.get("query", {}),
            "body": EvoBody.model_validate(body),
        })
    
    raise ValueError("payload shape not recognized")

# -------------------------
# JSON DB helpers
# -------------------------
def digits(s: str) -> str:
    """Extrai apenas dígitos de uma string"""
    return "".join(ch for ch in s if ch.isdigit())

def user_path(user_id: str) -> Path:
    """Retorna o caminho do arquivo JSON do usuário"""
    return DB_DIR / f"{digits(user_id)}.json"

def load_user(user_id: str) -> Dict[str, Any]:
    """Carrega dados do usuário do JSON"""
    p = user_path(user_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.exception(f"Falha ao ler JSON do usuário {user_id}: {e}")
    
    # Estrutura inicial
    return {
        "user_id": digits(user_id),
        "profile": {},
        "weights": [],
        "meals": [],
        "water": [],
        "workouts": [],
        "last_report": None,
    }

def save_user(user_id: str, data: Dict[str, Any]) -> None:
    """Salva dados do usuário no JSON"""
    p = user_path(user_id)
    try:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"✅ DB salvo: {p}")
    except Exception as e:
        log.error(f"❌ Erro ao salvar {p}: {e}")
        raise

# -------------------------
# Redis histórico de interação
# -------------------------
def push_history(user_id: str, who: str, text: str) -> None:
    """Adiciona mensagem ao histórico no Redis"""
    key = f"fit:history:{digits(user_id)}"
    item = json.dumps({
        "ts": datetime.utcnow().isoformat(), 
        "from": who, 
        "text": text
    }, ensure_ascii=False)
    
    try:
        r.lpush(key, item)
        r.ltrim(key, 0, 499)  # mantém últimas 500 mensagens
        log.info(f"HIST [{key}] <= {who}: {text[:100]}...")
    except Exception as e:
        log.error(f"Erro ao salvar histórico: {e}")

# -------------------------
# Parsers de comandos (WhatsApp)
# -------------------------
_kfloat = lambda s: float(str(s).replace(",", "."))

re_peso = re.compile(r"^\s*peso\s+(\d+[\.,]?\d*)", re.I)
re_agua = re.compile(r"^\s*(agua|água)\s+(\d+[\.,]?\d*)", re.I)
re_meal = re.compile(r"^\s*(?:refe[ií]cao|refe[ií]ção)", re.I)
re_treino = re.compile(r"^\s*treino\s+(?P<tipo>\w+)\s+(?P<dur>\d+)\s*min(?:\s+(?P<sets>\d+)x(?P<reps>\d+))?", re.I)
re_perfil = re.compile(r"perfil\s+altura\s+(?P<alt>\d{3})\s+sexo\s+(?P<sex>\w+)\s+objetivo\s+(?P<goal>\w+)\s+atividade\s+(?P<act>\w+)", re.I)

def parse_meal(text: str) -> Optional[Dict[str, Any]]:
    """Parser para refeições: refeicao 650kcal 35p 75c 15g descrição"""
    try:
        kcal = re.search(r"(\d+)\s*kcal", text, re.I)
        prot = re.search(r"(\d+)\s*p", text, re.I)
        carb = re.search(r"(\d+)\s*c", text, re.I)
        fat = re.search(r"(\d+)\s*g(?!r)", text, re.I)
        
        desc = re.split(r"kcal|p\b|c\b|g\b", text, flags=re.I)
        desc = desc[-1].strip() if desc else ""
        
        if kcal and prot and carb and fat:
            return {
                "kcal": int(kcal.group(1)),
                "protein_g": int(prot.group(1)),
                "carbs_g": int(carb.group(1)),
                "fat_g": int(fat.group(1)),
                "desc": desc[:100]
            }
    except Exception as e:
        log.error(f"Erro ao parsear refeição: {e}")
    
    return None

# -------------------------
# CrewAI — Configuração Simplificada
# -------------------------
def build_agents():
    """Cria agentes usando configuração simplificada que funciona com LiteLLM"""
    
    dietitian = Agent(
        role="Nutricionista IA",
        goal=f"Analisar ingestão e sugerir metas realistas de 7 dias no estilo {BRAND}.",
        backstory="Prática e clara; dá números que cabem no dia-a-dia.",
        model=MODEL,  # Usar diretamente o modelo da variável
        temperature=TEMPERATURE,
        allow_delegation=False, 
        verbose=False,
    )
    
    trainer = Agent(
        role="Treinador IA",
        goal="Avaliar volume de treino e propor progressão segura para 7 dias.",
        backstory="Consistência > intensidade; foco em técnica.",
        model=MODEL,
        temperature=TEMPERATURE,
        allow_delegation=False, 
        verbose=False,
    )
    
    habit = Agent(
        role="Coach de Hábitos",
        goal="Transformar metas em um checklist diário de 5 itens, com gatilhos de contexto.",
        backstory="Sem clichês; orientações curtas e acionáveis.",
        model=MODEL,
        temperature=TEMPERATURE,
        allow_delegation=False, 
        verbose=False,
    )
    
    chef = Agent(
        role="Chef Saudável",
        goal="Sugerir 3 receitas (café, almoço, jantar) com macros aproximados.",
        backstory="Ingredientes comuns no Brasil; preparo rápido.",
        model=MODEL,
        temperature=TEMPERATURE,
        allow_delegation=False, 
        verbose=False,
    )
    
    router = Agent(
        role="Coach Conversacional",
        goal="Conversar naturalmente sobre saúde, fitness e bem-estar, sendo empático e engajador.",
        backstory="Coach amigável que adapta linguagem ao usuário, faz perguntas relevantes e motiva de forma natural.",
        model=MODEL,
        temperature=TEMPERATURE,
        allow_delegation=False, 
        verbose=False,
    )
    
    return dietitian, trainer, habit, chef, router

def build_coach_tasks(profile: Dict[str, Any], week: Dict[str, List[Dict[str, Any]]]):
    """Constrói as tasks para análise completa"""
    ctx = json.dumps({"profile": profile, **week}, ensure_ascii=False)
    diet, trn, hab, chf, _ = build_agents()
    
    t1 = Task(
        description=(
            "Dados de 7 dias: peso, refeições (kcal, macros) e água.\n"
            "1) Faça média diária de kcal e macros; 2) compare com objetivo (cut/maintain/gain) e perfil;\n"
            "3) defina metas para a próxima semana (kcal e proteína) + alvo de água (L).\n"
            f"CONTEXT: {ctx}"
        ),
        agent=diet,
        expected_output="Metas numéricas e breve justificativa."
    )
    
    t2 = Task(
        description=(
            "Com base nos treinos, proponha plano de 7 dias com 3 sessões exemplo (duração, séries×reps e RPE).\n"
            f"CONTEXT: {ctx}"
        ),
        agent=trn,
        expected_output="Plano semanal objetivo."
    )
    
    t3 = Task(
        description=(
            "Converta metas em um checklist diário (5 itens) e 3 alavancas de adesão.\n"
            f"CONTEXT: {ctx}"
        ),
        agent=hab,
        expected_output="Checklist e hábitos."
    )
    
    t4 = Task(
        description=(
            "Sugira 3 receitas simples (café, almoço, jantar) com macros aproximados por porção.\n"
            f"CONTEXT: {ctx}"
        ),
        agent=chf,
        expected_output="3 receitas com passos curtos."
    )
    
    return [t1, t2, t3, t4]

async def run_coach(profile: Dict[str, Any], week: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Executa análise completa com CrewAI"""
    try:
        log.info("🚀 Iniciando análise CrewAI...")
        tasks = build_coach_tasks(profile, week)
        agents = list({t.agent for t in tasks})
        crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)
        
        res = await crew.kickoff_async()
        
        out: Dict[str, Any] = {}
        for i, t in enumerate(tasks, start=1):
            raw = getattr(t.output, 'raw', None)
            out[f"task_{i}"] = raw if raw else str(t.output)
        
        log.info("✅ Coach executado com sucesso")
        return out
        
    except Exception as e:
        log.error(f"❌ Erro no CrewAI: {e}")
        return {
            "error": True,
            "message": f"Erro na análise: {str(e)}",
            "task_1": "🎯 Mantenha suas metas atuais de calorias e continue hidratando bem!",
            "task_2": "💪 Continue com treinos regulares de 3-4x por semana, 45-60 min cada.",
            "task_3": "🔄 Foque em: 1) Dormir 7-8h 2) Beber 2L água 3) 3 refeições balanceadas 4) 30min movimento 5) Descanso ativo",
            "task_4": "👨‍🍳 Sugestões: Café: aveia com frutas | Almoço: arroz, feijão, proteína, salada | Jantar: proteína leve com vegetais"
        }

# -------------------------
# WhatsApp Sender - CORRIGIDO PARA "to"
# -------------------------
async def send_whatsapp(number: str, text: str) -> Dict[str, Any]:
    """Envia mensagem via WABot API"""
    url = f"{WABOT_BASE}{WABOT_SEND_PATH}"
    headers = {"Content-Type": "application/json", "x-api-key": WABOT_API_KEY}
    payload = {"to": digits(number), "text": text}  # ← CORRIGIDO: "to" em vez de "number"
    
    log.info(f"📱 Enviando WhatsApp → {payload['to']} | {text[:120]}...")
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        log.error(f"❌ sendText falhou: {e} | body={r.text}")
        raise HTTPException(status_code=502, detail=f"sendText error: {e}")
    except Exception as e:
        log.error(f"❌ Erro ao enviar WhatsApp: {e}")
        return {"status": "error", "message": str(e)}

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title=f"{BRAND} — Wellness WhatsApp (CrewAI+Redis+JSON)")

@app.get("/")
async def index():
    return {
        "ok": True,
        "service": "Wellness WhatsApp",
        "model": MODEL,
        "routes": ["/health", "/whatsapp/webhook (POST)", "/profile, /log/*, /summary/*, /agent/*", "/docs"]
    }

@app.head("/")
async def index_head():
    return Response(status_code=200)

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/whatsapp/webhook")
async def whatsapp_webhook_get():
    return JSONResponse({"ok": True, "hint": "Use POST neste mesmo URL com payload message.received"}, status_code=200)

@app.get("/health")
async def health():
    """Health check com status do Redis e OpenRouter"""
    try:
        redis_ok = r.ping()
    except Exception:
        redis_ok = False
    
    users = len(list(DB_DIR.glob("*.json")))
    
    return {
        "ok": True,
        "model": MODEL,
        "brand": BRAND,
        "users": users,
        "redis": redis_ok,
        "openrouter": bool(OPENROUTER_API_KEY),
        "timestamp": datetime.utcnow().isoformat()
    }

# -------------------------
# REST endpoints simplificados
# -------------------------
class Profile(BaseModel):
    user_id: str
    height_cm: Optional[int] = Field(None, ge=100, le=250)
    sex: Optional[str] = Field(None, description="male|female|other")
    goal: Optional[str] = Field(None, description="cut|maintain|gain")
    activity: Optional[str] = Field(None, description="sedentary|light|moderate|high|athlete")

def _today(s: Optional[str]) -> str:
    return (s or datetime.utcnow().date().isoformat())[:10]

@app.post("/profile")
async def set_profile(p: Profile):
    u = load_user(p.user_id)
    for k in ("height_cm","sex","goal","activity"):
        v = getattr(p, k)
        if v is not None:
            u.setdefault("profile", {})[k] = v
    save_user(p.user_id, u)
    return {"ok": True}

@app.get("/summary/{user_id}")
async def summary(user_id: str, days: int = Query(7, ge=1, le=31)):
    u = load_user(user_id)
    start = (datetime.utcnow().date() - timedelta(days=days-1)).isoformat()
    end = datetime.utcnow().date().isoformat()
    
    def filt(items: List[Dict[str, Any]]):
        return [x for x in items if start <= x.get("date","")[:10] <= end]
    
    weights = filt(u.get("weights", []))
    meals = filt(u.get("meals", []))
    water = filt(u.get("water", []))
    workouts = filt(u.get("workouts", []))
    
    return {
        "range": {"start": start, "end": end},
        "entries": {"weights": len(weights), "meals": len(meals), "water": len(water), "workouts": len(workouts)},
        "totals": {
            "kcal": sum(m.get('kcal', 0) for m in meals),
            "protein_g": sum(m.get('protein_g', 0) for m in meals),
            "water_l": round(sum(w.get('liters', 0.0) for w in water), 2)
        },
        "latest_weight": weights[0] if weights else None,
    }

@app.post("/agent/coach/{user_id}")
async def coach(user_id: str, days: int = Query(7, ge=3, le=31)):
    u = load_user(user_id)
    profile = u.get("profile", {}) or {"goal": "maintain", "activity": "moderate"}
    
    start = (datetime.utcnow().date() - timedelta(days=days-1)).isoformat()
    
    def filt(items: List[Dict[str, Any]]):
        return [x for x in items if start <= x.get("date","")[:10]]
    
    data = {
        "weights": filt(u.get("weights", [])),
        "meals": filt(u.get("meals", [])),
        "water": filt(u.get("water", [])),
        "workouts": filt(u.get("workouts", [])),
    }
    
    report = await run_coach(profile=profile, week=data)
    u["last_report"] = report
    save_user(user_id, u)
    
    return {"ok": True, "range": {"start": start}, "report": report}

# -------------------------
# WhatsApp Webhook - CONVERSAÇÃO NATURAL
# -------------------------
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, payload: dict | list = Body(...)):
    ip = request.client.host if request.client else "?"
    log.info(f"📥 Webhook de {ip}: recebido")
    
    try:
        item = payload[0] if isinstance(payload, list) else payload
        if isinstance(item, dict) and item.get("body"):
            obj = {
                "headers": item.get("headers", {}),
                "params": item.get("params", {}),
                "query": item.get("query", {}),
                "body": item.get("body", {}),
            }
        else:
            body = item if isinstance(item, dict) else {}
            if "event" not in body and "data" in body:
                body["event"] = "message.received"
            obj = {"headers": {}, "params": {}, "query": {}, "body": body}
        
        env = EvoEnvelope.model_validate(obj)
    except Exception as e:
        log.error(f"❌ Payload inválido: {e}")
        raise HTTPException(status_code=400, detail=f"payload inválido: {e}")
    
    hdr_evt = (env.headers or {}).get("x-webhook-event", "")
    body_evt = env.body.event
    
    if hdr_evt != "message.received" and body_evt != "message.received":
        return {"ignored": True, "reason": "not a message.received"}
    
    data = env.body.data
    
    if data.isGroup:
        return {"ignored": True, "reason": "group message"}
    
    if data.type != "text" or not (data.text or "").strip():
        return {"ignored": True, "reason": "non-text or empty"}
    
    user_id = digits(data.sender or data.jid)
    text = (data.text or "").strip()
    log.info(f"✅ Normalizado: number={user_id} text={text!r}")
    
    push_history(user_id, "user", text)
    
    # Processar comandos
    reply: Optional[str] = None
    u = load_user(user_id)
    
    # PERFIL
    m = re_perfil.search(text)
    if m:
        u.setdefault("profile", {})
        u["profile"].update({
            "height_cm": int(m.group("alt")),
            "sex": m.group("sex"),
            "goal": m.group("goal"),
            "activity": m.group("act"),
        })
        save_user(user_id, u)
        reply = "✅ Perfil atualizado! Agora posso dar dicas mais personalizadas. Quer fazer uma análise completa? Digite 'coach' 😊"
    
    # PESO
    if not reply:
        m = re_peso.search(text)
        if m:
            item = {"date": datetime.utcnow().date().isoformat(), "kg": _kfloat(m.group(1))}
            u.setdefault("weights", []).insert(0, item)
            save_user(user_id, u)
            reply = f"⚖️ Perfeito! Peso de {item['kg']} kg registrado. Como você está se sentindo hoje?"
    
    # ÁGUA
    if not reply:
        m = re_agua.search(text)
        if m:
            item = {"date": datetime.utcnow().date().isoformat(), "liters": _kfloat(m.group(2))}
            u.setdefault("water", []).insert(0, item)
            save_user(user_id, u)
            reply = f"💧 Ótimo! {item['liters']} L de água registrados. Hidratação é fundamental! 👏"
    
    # REFEIÇÃO
    if not reply and re_meal.search(text):
        meal = parse_meal(text)
        if meal:
            meal["date"] = datetime.utcnow().date().isoformat()
            u.setdefault("meals", []).insert(0, meal)
            save_user(user_id, u)
            reply = f"🍽️ Refeição registrada! {meal['kcal']} kcal com {meal['protein_g']}g de proteína. Que tal registrar a próxima também?"
        else:
            reply = "❌ Não consegui entender o formato da refeição. Use: refeicao 650kcal 35p 75c 15g omelete com aveia"
    
    # TREINO
    if not reply:
        m = re_treino.search(text)
        if m:
            item = {
                "date": datetime.utcnow().date().isoformat(),
                "type": m.group("tipo"),
                "duration_min": int(m.group("dur")),
                "sets": int(m.group("sets")) if m.group("sets") else None,
                "reps": int(m.group("reps")) if m.group("reps") else None,
                "notes": "",
            }
            u.setdefault("workouts", []).insert(0, item)
            save_user(user_id, u)
            reply = f"💪 Parabéns pelo treino! {item['type'].title()} por {item['duration_min']} minutos. Como foi a intensidade?"
    
    # COACH
    if not reply and re.search(r"\b(coach|análise|analise|analisar)\b", text, re.I):
        profile = u.get("profile", {}) or {"goal": "maintain", "activity": "moderate"}
        data_week = {
            "weights": u.get("weights", [])[:20],
            "meals": u.get("meals", [])[:30],
            "water": u.get("water", [])[:20],
            "workouts": u.get("workouts", [])[:20],
        }
        
        log.info(f"🤖 Executando análise CrewAI para {user_id}...")
        report = await run_coach(profile, data_week)
        u["last_report"] = report
        save_user(user_id, u)
        
        # Compacta resposta para WhatsApp
        reply = (
            "🤖 *Sua análise personalizada está pronta!*\n\n" +
            "🎯 *Nutrição:*\n" + (report.get("task_1", "").strip()[:450]) + "\n\n" +
            "💪 *Treinos:*\n" + (report.get("task_2", "").strip()[:400]) + "\n\n" +
            "🔄 *Hábitos:*\n" + (report.get("task_3", "").strip()[:350]) + "\n\n" +
            "👨‍🍳 *Receitas:*\n" + (report.get("task_4", "").strip()[:300])
        )
    
    # CONVERSAÇÃO NATURAL COM IA
    if not reply:
        try:
            # Usar o agent router para conversas naturais
            diet, trn, hab, chf, router = build_agents()
            
            # Context sobre o usuário
            profile = u.get("profile", {})
            recent_data = {
                "peso_recente": u.get("weights", [{}])[0].get("kg") if u.get("weights") else None,
                "refeicoes_hoje": len([m for m in u.get("meals", []) if m.get("date") == datetime.utcnow().date().isoformat()]),
                "treinos_semana": len([w for w in u.get("workouts", []) if (datetime.utcnow().date() - datetime.fromisoformat(w.get("date", "1900-01-01"))).days <= 7])
            }
            
            context = f"Usuário: {text}\nPerfil: {profile}\nDados recentes: {recent_data}"
            
            t = Task(
                description=(
                    "Você é um coach de saúde amigável e conversacional pelo WhatsApp.\n"
                    "Responda de forma natural e humana. Se perguntarem como você funciona, explique que:\n"
                    "- Ajuda com nutrição, treinos e hábitos saudáveis\n"
                    "- Registra dados como peso, água, refeições e treinos\n"
                    "- Faz análises personalizadas com 'coach'\n"
                    "- Conversa naturalmente sobre saúde e bem-estar\n\n"
                    "Seja empático, use emojis moderadamente e faça perguntas para engajar.\n"
                    f"Contexto: {context}"
                ),
                agent=router,
                expected_output="Resposta conversacional, natural e engajadora (máximo 200 caracteres)"
            )
            
            c = Crew(agents=[router], tasks=[t], process=Process.sequential, verbose=False)
            await c.kickoff_async()
            
            ai_response = getattr(t.output, 'raw', None) or str(t.output)
            reply = ai_response.strip()
            
            # Fallback caso a IA falhe
            if not reply or len(reply) < 10:
                raise Exception("IA response too short")
                
        except Exception as e:
            log.error(f"Erro na conversa IA: {e}")
            # Fallback mais natural
            if re.search(r"\b(help|ajuda|como|usar|funciona|faz)\b", text, re.I):
                reply = (
                    "Sou seu coach de saúde! 💪\n\n"
                    "Posso ajudar você a:\n"
                    "• Acompanhar peso e hidratação\n"
                    "• Registrar refeições e treinos\n"
                    "• Fazer análises personalizadas\n\n"
                    "Experimente: peso 75 ou coach\n"
                    "Como posso te ajudar hoje? 😊"
                )
            else:
                reply = (
                    "Oi! 👋 Sou especialista em saúde e fitness.\n\n"
                    "Posso te ajudar com nutrição, treinos e hábitos saudáveis!\n\n"
                    "Quer começar registrando seu peso atual ou tem alguma pergunta sobre saúde? 🤔"
                )
    
    push_history(user_id, "coach", reply)
    await send_whatsapp(user_id, reply)
    
    return {"ok": True}

@app.post("/seed/{user_id}")
async def seed(user_id: str):
    """Popula dados de teste"""
    from random import randint
    u = load_user(user_id)
    u["profile"] = {"height_cm": 178, "sex": "male", "goal": "cut", "activity": "moderate"}
    
    today = datetime.utcnow().date()
    for d in range(3):
        date = (today - timedelta(days=d)).isoformat()
        u.setdefault("weights", []).insert(0, {"date": date, "kg": 82.0 - d*0.2})
        u.setdefault("water", []).insert(0, {"date": date, "liters": 2.2 + 0.1*d})
        
        for _ in range(2):
            u.setdefault("meals", []).insert(0, {
                "date": date, "kcal": randint(550, 850),
                "protein_g": randint(25, 45), "carbs_g": randint(60, 110), "fat_g": randint(10, 25),
                "desc": "refeição demo"
            })
        
        u.setdefault("workouts", []).insert(0, {
            "date": date, "type": "força", "duration_min": 45, "reps": 10, "sets": 4, "notes": "treino demo"
        })
    
    save_user(user_id, u)
    return {"ok": True, "message": "Dados de demonstração criados"}

if __name__ == "__main__":
    import uvicorn
    log.info("🚀 Iniciando servidor...")
    uvicorn.run("wellness_agent_whatsapp_json_redis:app", host="0.0.0.0", port=8000, reload=True)

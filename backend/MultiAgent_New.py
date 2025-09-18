"""
Voice Chatbot Call Center Multi-Agent System with CrewAI
Fast hierarchical architecture with RAG and Google Sheets integration
"""

from __future__ import annotations

import os
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
import pandas as pd
import pytz
from collections import OrderedDict
from pydantic import BaseModel, Field

# Core CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    try:
        from crewai import LLM
    except Exception:
        from crewai.llm import LLM
    from crewai.tools import BaseTool
except ImportError as e:
    raise ImportError("crewai is required. Install with: pip install crewai litellm") from e

# --- Compatibility monkeypatch: make AskQuestion tool schema more forgiving
# Some external tool schemas (AskQuestionToolSchema) can be strict and raise
# pydantic validation errors when callers omit optional context/coworker fields.
# If crewai is available at runtime, replace or define a permissive schema so
# missing 'context' or 'coworker' won't crash the workflow.
try:
    import crewai
    # Only attempt this if the tools module exists
    if hasattr(crewai, 'tools'):
        try:
            # Create a permissive schema model
            class _PermissiveAskQuestionSchema(BaseModel):
                question: str
                context: Optional[str] = None
                coworker: Optional[str] = None

            # Inject into crewai.tools if not present or replace existing strict schema
            setattr(crewai.tools, 'AskQuestionToolSchema', _PermissiveAskQuestionSchema)
            setattr(crewai, 'AskQuestionToolSchema', _PermissiveAskQuestionSchema)
            # Also try to update any AskQuestion tool class present in crewai.tools
            for name in dir(crewai.tools):
                try:
                    obj = getattr(crewai.tools, name)
                    if isinstance(obj, type) and 'Ask' in name:
                        # if it has an args_schema attr, replace it
                        if hasattr(obj, 'args_schema'):
                            setattr(obj, 'args_schema', _PermissiveAskQuestionSchema)
                except Exception:
                    pass
            print("Applied permissive AskQuestionToolSchema to crewai.tools (best-effort)")
        except Exception as _e:
            print(f"Could not patch AskQuestionToolSchema: {_e}")
except Exception:
    # If crewai isn't importable in this environment, skip patching silently
    pass

# Optional imports for full functionality
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    print("Warning: Google Sheets integration not available. Install: pip install gspread oauth2client")

try:
    import PyPDF2
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG functionality not available. Install: pip install PyPDF2 faiss-cpu sentence-transformers")

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Configuration constants
TOGETHER_MODEL = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"
OPENROUTER_MODEL = "openrouter/qwen/qwen3-4b:free"

# Mock Product Database
MOCK_PRODUCTS = [
    {
        "sku": "TEL001",
        "product_name": "สมาร์ทโฟน Galaxy A54",
        "category": "โทรศัพท์",
        "owner_name": "นายสมชาย ใจดี",
        "returned_location": "ร้านค้า Central",
        "returned_date": "2024-12-01"
    },
    {
        "sku": "INT002", 
        "product_name": "แพ็กเกจอินเทอร์เน็ต Fiber 100/30",
        "category": "อินเทอร์เน็ต",
        "owner_name": "นางสาวมณี สีทอง",
        "returned_location": "สาขาเซ็นทรัล",
        "returned_date": "2024-11-28"
    },
    {
        "sku": "TV003",
        "product_name": "Smart TV Samsung 55 นิ้ว",
        "category": "เครื่องใช้ไฟฟ้า",
        "owner_name": "นายประชา รักดี",
        "returned_location": "โลตัส สาขา 2",
        "returned_date": "2024-12-05"
    }
]

def _late_env_hydrate():
    """Load .env file if API keys not found."""
    if os.getenv("TOGETHER_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
        return
    
    current = os.path.dirname(__file__)
    search_paths = [current, os.path.dirname(current)]
    
    for base_path in search_paths:
        for _ in range(6):
            env_path = os.path.join(base_path, '.env')
            if os.path.exists(env_path):
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if not s or s.startswith('#') or '=' not in s:
                                continue
                            k, v = s.split('=', 1)
                            k = k.strip(); v = v.strip().strip('"').strip("'")
                            if k and v and k not in os.environ:
                                os.environ[k] = v
                    if os.getenv("TOGETHER_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
                        return
                except Exception as e:
                    logger.debug(f"Error reading {env_path}: {e}")
            parent = os.path.dirname(base_path)
            if parent == base_path:
                break
            base_path = parent


# =============================================================================
# PYDANTIC INPUT SCHEMAS
# =============================================================================

class ProductQueryInput(BaseModel):
    """Input schema for product database queries."""
    query_type: str = Field(..., description="Type of query: 'search', 'get_by_sku', 'get_by_owner', 'list_all'")
    search_term: Optional[str] = Field(None, description="Search term for product name, category, or owner")
    sku: Optional[str] = Field(None, description="Product SKU to lookup")
    owner_name: Optional[str] = Field(None, description="Owner name to search by")

class SheetsQueryInput(BaseModel):
    """Input schema for Google Sheets operations."""
    spreadsheet_name: str = Field(..., description="Name of the Google Sheet")
    operation: str = Field(..., description="Operation: 'read', 'search', 'add_row'")
    search_query: Optional[str] = Field(None, description="Search query for finding specific data")
    new_row_data: Optional[dict] = Field(None, description="Data for new row as dict")

class RAGQueryInput(BaseModel):
    """Input schema for RAG PDF queries."""
    query: str = Field(..., description="Natural language query to search in PDF documents")
    top_k: Optional[int] = Field(default=3, description="Number of top results to return")


# =============================================================================
# TOOLS IMPLEMENTATION
# =============================================================================

class TimeTool(BaseTool):
    """Tool for getting current Thai time."""
    name: str = "time_tool"
    description: str = "Get current date and time in Thailand timezone"
    
    def _run(self, query: str = "") -> str:
        thailand_tz = pytz.timezone('Asia/Bangkok')
        current_time = datetime.now(thailand_tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"เวลาปัจจุบันในประเทศไทย: {current_time}"


class ProductDatabaseTool(BaseTool):
    """Tool for querying mock product database."""
    name: str = "product_database_tool"
    description: str = "Query product database for SKU, product names, categories, owners, and return information"
    args_schema: Type[ProductQueryInput] = ProductQueryInput
    
    def _run(self, query_type: str, search_term: Optional[str] = None, 
             sku: Optional[str] = None, owner_name: Optional[str] = None) -> str:
        
        if query_type == "list_all":
            return f"พบสินค้าทั้งหมด {len(MOCK_PRODUCTS)} รายการ:\n" + \
                   "\n".join([f"- {p['sku']}: {p['product_name']} ({p['category']})" 
                             for p in MOCK_PRODUCTS])
        
        elif query_type == "get_by_sku" and sku:
            for product in MOCK_PRODUCTS:
                if product['sku'].upper() == sku.upper():
                    return f"พบสินค้า: {product['product_name']} (รหัส: {product['sku']})\n" + \
                           f"หมวดหมู่: {product['category']}\n" + \
                           f"เจ้าของ: {product['owner_name']}\n" + \
                           f"สถานที่คืน: {product['returned_location']}\n" + \
                           f"วันที่คืน: {product['returned_date']}"
            return f"ไม่พบสินค้าที่มีรหัส {sku}"
        
        elif query_type == "get_by_owner" and owner_name:
            results = [p for p in MOCK_PRODUCTS if owner_name.lower() in p['owner_name'].lower()]
            if results:
                return "พบสินค้าของ " + owner_name + ":\n" + \
                       "\n".join([f"- {p['sku']}: {p['product_name']}" for p in results])
            return f"ไม่พบสินค้าของ {owner_name}"
        
        elif query_type == "search" and search_term:
            results = []
            term = search_term.lower()
            for product in MOCK_PRODUCTS:
                if (term in product['product_name'].lower() or 
                    term in product['category'].lower() or
                    term in product['owner_name'].lower()):
                    results.append(product)
            
            if results:
                return f"พบ {len(results)} รายการ:\n" + \
                       "\n".join([f"- {p['sku']}: {p['product_name']} ({p['category']})" 
                                 for p in results])
            return f"ไม่พบสินค้าที่ตรงกับ '{search_term}'"
        
        return "กรุณาระบุพารามิเตอร์ที่ถูกต้อง"


class GoogleSheetsTool(BaseTool):
    """Tool for Google Sheets operations."""
    name: str = "google_sheets_tool"
    description: str = "Read, search, and add data to Google Sheets"
    args_schema: Type[SheetsQueryInput] = SheetsQueryInput
    # Declare client as a field so pydantic/BaseTool allows assignment in __init__
    client: Optional[Any] = None
    # Simple LRU cache for recent queries to speed repeated lookups
    cache: Dict[str, str] = Field(default_factory=dict)
    cache_max: int = 128
    
    def __init__(self):
        super().__init__()
        self.client = None
        # instance cache (OrderedDict for simple LRU behavior)
        self.cache = OrderedDict()
        self.cache_max = 128
        if SHEETS_AVAILABLE:
            try:
                scope = ['https://spreadsheets.google.com/feeds', 
                        'https://www.googleapis.com/auth/drive']
                credentials = ServiceAccountCredentials.from_json_keyfile_name(
                    'client_secret.json', scope)
                self.client = gspread.authorize(credentials)
            except FileNotFoundError:
                logger.warning("Google Sheets credentials not found")
    
    def _run(self, spreadsheet_name: str, operation: str, 
             search_query: Optional[str] = None, new_row_data: Optional[dict] = None) -> str:
        
        # Cache key for speed
        cache_key = f"sheets:{spreadsheet_name}|{operation}|{search_query}|{json.dumps(new_row_data, ensure_ascii=False) if new_row_data else ''}"
        if cache_key in getattr(self, "cache", {}):
            return self.cache[cache_key]

        if not SHEETS_AVAILABLE or self.client is None:
            # Mock response for demo purposes
            if operation == "search" and search_query:
                resp = f"Mock: พบข้อมูลที่ตรงกับ '{search_query}' ในสเปรดชีต {spreadsheet_name}"
                # store cache
                self.cache[cache_key] = resp
                while len(self.cache) > self.cache_max:
                    self.cache.popitem(last=False)
                return resp
            elif operation == "add_row" and new_row_data:
                resp = f"Mock: เพิ่มข้อมูลใหม่ในสเปรดชีต {spreadsheet_name} เรียบร้อย"
                self.cache[cache_key] = resp
                while len(self.cache) > self.cache_max:
                    self.cache.popitem(last=False)
                return resp
            elif operation == "read":
                resp = f"Mock: อ่านข้อมูลจากสเปรดชีต {spreadsheet_name} เรียบร้อย"
                self.cache[cache_key] = resp
                while len(self.cache) > self.cache_max:
                    self.cache.popitem(last=False)
                return resp
            return "Mock: Google Sheets ไม่พร้อมใช้งาน"
        
        try:
            spreadsheet = self.client.open(spreadsheet_name)
            worksheet = spreadsheet.sheet1
            
            if operation == "read":
                data = worksheet.get_all_records()
                resp = f"อ่านข้อมูล {len(data)} แถวจากสเปรดชีต {spreadsheet_name}"
                self.cache[cache_key] = resp
                while len(self.cache) > self.cache_max:
                    self.cache.popitem(last=False)
                return resp
            
            elif operation == "search" and search_query:
                data = worksheet.get_all_records()
                # Simple search implementation
                results = []
                for row in data:
                    for value in row.values():
                        if search_query.lower() in str(value).lower():
                            results.append(row)
                            break
                resp = f"พบ {len(results)} รายการที่ตรงกับ '{search_query}'"
                self.cache[cache_key] = resp
                while len(self.cache) > self.cache_max:
                    self.cache.popitem(last=False)
                return resp
            
            elif operation == "add_row" and new_row_data:
                values = list(new_row_data.values())
                worksheet.append_row(values)
                resp = f"เพิ่มข้อมูลใหม่ในสเปรดชีต {spreadsheet_name} เรียบร้อย"
                self.cache[cache_key] = resp
                while len(self.cache) > self.cache_max:
                    self.cache.popitem(last=False)
                return resp
            
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
        
        return "ไม่สามารถดำเนินการได้"


class RAGTool(BaseTool):
    """Tool for RAG-based PDF document queries."""
    name: str = "rag_tool"
    description: str = "Search and retrieve information from PDF documents using RAG"
    args_schema: Type[RAGQueryInput] = RAGQueryInput
    # Pydantic fields for attributes assigned during initialization
    embedder: Optional[Any] = None
    index: Optional[Any] = None
    chunks: List[str] = Field(default_factory=list)
    
    def __init__(self):
        super().__init__()
        self.embedder = None
        self.index = None
        self.chunks = []
        
        if RAG_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                # Mock some Thai content for demo
                self.chunks = [
                    "นโยบายการคืนสินค้า: สินค้าสามารถคืนได้ภายใน 30 วัน",
                    "การรับประกันสินค้า: สมาร์ทโฟนมีการรับประกัน 1 ปี",
                    "ค่าบริการอินเทอร์เน็ต: แพ็กเกจ Fiber เริ่มต้น 590 บาทต่อเดือน",
                    "ขั้นตอนการติดตั้งอุปกรณ์: ช่างจะติดนัดติดตั้งภายใน 3-5 วันทำการ"
                ]
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(embeddings.astype('float32'))
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}")
    
    def _run(self, query: str, top_k: int = 3) -> str:
        if not RAG_AVAILABLE or self.embedder is None:
            # Mock response
            return f"Mock RAG: พบข้อมูลเกี่ยวกับ '{query}' ในเอกสาร PDF - การรับประกันสินค้า 1 ปี, นโยบายการคืนสินค้าภายใน 30 วัน"
        
        try:
            query_embedding = self.embedder.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if scores[0][i] > 0.3:  # Similarity threshold
                    results.append(self.chunks[idx])
            
            if results:
                return f"พบข้อมูลที่เกี่ยวข้อง:\n" + "\n".join([f"- {r}" for r in results])
            else:
                return f"ไม่พบข้อมูลที่เกี่ยวข้องกับ '{query}'"
                
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}"


# =============================================================================
# MULTIAGENT CONFIGURATION
# =============================================================================

@dataclass
class VoiceCallCenterConfig:
    """Configuration for voice call center system."""
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", TOGETHER_MODEL))
    temperature: float = 0.2  # Lower for more consistent responses
    max_tokens: int = 128  # Shorter responses for speed
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    request_timeout: int = 30  # Faster timeout
    # Speed-mode options
    speed_mode: bool = True  # When True, prefer faster responses over quality
    fast_model: Optional[str] = None  # Optional cheaper/faster model override
    
    def resolve(self):
        # Try OpenRouter first, then Together AI
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if self.api_key:
                self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                if self.model == TOGETHER_MODEL:
                    self.model = OPENROUTER_MODEL
        
        if not self.api_key:
            self.api_key = os.getenv("TOGETHER_API_KEY")
            if self.api_key:
                self.base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        
        if not self.api_key:
            _late_env_hydrate()
            self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("TOGETHER_API_KEY")
            if os.getenv("OPENROUTER_API_KEY"):
                self.base_url = "https://openrouter.ai/api/v1"
                if self.model == TOGETHER_MODEL:
                    self.model = OPENROUTER_MODEL
            else:
                self.base_url = "https://api.together.xyz/v1"
        
        return self

    def refresh(self):
        """Refresh environment-derived settings.

        Kept for compatibility with callers that expect a `refresh()` method
        (e.g. older server code calling `self.config.refresh()`). This simply
        re-runs resolve() and returns self.
        """
        # Re-run resolution in case environment variables or .env changed
        try:
            return self.resolve()
        except Exception:
            # Resolve should be safe; if it fails, return current config
            return self


# =============================================================================
# MAIN VOICE CALL CENTER SYSTEM
# =============================================================================

class VoiceCallCenterMultiAgent:
    """
    Fast hierarchical multi-agent system for voice call center.
    
    Architecture:
    - Supervisor Agent: Routes requests to specialized agents
    - Product Agent: Handles product queries via database/sheets
    - Knowledge Agent: Handles policy/procedure queries via RAG
    - Response Agent: Composes final customer responses
    
    Usage:
        system = VoiceCallCenterMultiAgent()
        result = system.process_voice_input("สวัสดีครับ ขอสอบถามแพ็กเกจอินเทอร์เน็ต")
        print(result["response"])
    """
    
    def __init__(self, config: Optional[VoiceCallCenterConfig] = None):
        _late_env_hydrate()
        
        self.config = (config or VoiceCallCenterConfig()).resolve()
        if not self.config.api_key:
            raise RuntimeError("Missing TOGETHER_API_KEY or OPENROUTER_API_KEY")
        
        # Set environment variables for LiteLLM
        if "openrouter.ai" in (self.config.base_url or ""):
            os.environ["OPENROUTER_API_KEY"] = self.config.api_key
        else:
            os.environ["TOGETHER_API_KEY"] = self.config.api_key
        
        # Apply speed-mode overrides
        effective_model = self.config.model
        effective_max_tokens = self.config.max_tokens
        effective_timeout = self.config.request_timeout
        effective_temperature = self.config.temperature
        if getattr(self.config, "speed_mode", False):
            # Adjust for faster responses
            effective_max_tokens = min(64, int(effective_max_tokens))
            effective_timeout = max(5, int(effective_timeout // 2))
            effective_temperature = max(0.0, float(effective_temperature))
            if self.config.fast_model:
                effective_model = self.config.fast_model

        # Initialize LLM
        self.llm = LLM(
            model=effective_model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            timeout=effective_timeout,
        )
        
        # Initialize tools
        self.tools = {
            'time': TimeTool(),
            'products': ProductDatabaseTool(),
            'sheets': GoogleSheetsTool(),
            'rag': RAGTool()
        }

        # Cache agents to avoid recreating them on every request (reduces overhead)
        self._cached_agents: Dict[str, Agent] = {}
        
        logger.info(f"VoiceCallCenter initialized | model={self.config.model} | tools={list(self.tools.keys())}")
    
    def _create_supervisor_agent(self) -> Agent:
        """Create the supervisor agent that routes requests."""
        if 'supervisor' in self._cached_agents:
            return self._cached_agents['supervisor']

        agent = Agent(
            role="Call Center Supervisor",
            goal="รับฟังคำถามลูกค้าและส่งต่อไปยังผู้เชี่ยวชาญที่เหมาะสม อย่างรวดเร็วและแม่นยำ",
            backstory=(
                "คุณเป็นหัวหน้าทีม Call Center ที่มีประสบการณ์สูง "
                "สามารถแยกแยะประเภทคำถามและส่งต่อให้ผู้เชี่ยวชาญได้อย่างรวดเร็ว "
                "เช่น: สินค้า -> Product Agent, นโยบาย -> Knowledge Agent"
            ),
            llm=self.llm,
            allow_delegation=True,
            verbose=False,
            max_iter=2,  # Limit iterations for speed
        )
        self._cached_agents['supervisor'] = agent
        return agent
    
    def _create_product_agent(self) -> Agent:
        """Create the product specialist agent."""
        if 'product' in self._cached_agents:
            return self._cached_agents['product']

        agent = Agent(
            role="Product Specialist",
            goal="ค้นหาและให้ข้อมูลสินค้า SKU รายละเอียดการคืนสินค้า และข้อมูลเจ้าของสินค้าอย่างรวดเร็ว",
            backstory=(
                "คุณเป็นผู้เชี่ยวชาญด้านสินค้าและฐานข้อมูลลูกค้า "
                "มีความเชี่ยวชาญในการค้นหาข้อมูลสินค้า การคืนสินค้า และรายละเอียดการจัดส่ง"
            ),
            tools=[self.tools['products'], self.tools['sheets'], self.tools['time']],
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
    )
        self._cached_agents['product'] = agent
        return agent
    
    def _create_knowledge_agent(self) -> Agent:
        """Create the knowledge specialist agent."""
        if 'knowledge' in self._cached_agents:
            return self._cached_agents['knowledge']

        agent = Agent(
            role="Knowledge Specialist", 
            goal="ค้นหาข้อมูลจากเอกสารนโยบาย คู่มือ และให้คำแนะนำตามระเบียบบริษัทอย่างแม่นยำ",
            backstory=(
                "คุณเป็นผู้เชี่ยวชาญด้านนโยบายและระเบียบบริษัท "
                "มีความรู้เกี่ยวกับการรับประกัน การคืนสินค้า และขั้นตอนต่างๆ ของบริษัท"
            ),
            tools=[self.tools['rag'], self.tools['time']],
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
    )
        self._cached_agents['knowledge'] = agent
        return agent
    
    def _create_response_agent(self) -> Agent:
        """Create the response composition agent."""
        if 'response' in self._cached_agents:
            return self._cached_agents['response']

        agent = Agent(
            role="Customer Response Specialist",
            goal="จัดทำคำตอบที่สุภาพ กระชับ และนำไปใช้ได้ทันที สำหรับลูกค้าผ่านระบบ Voice Chat",
            backstory=(
                "คุณเป็นผู้เชี่ยวชาญด้านการสื่อสารกับลูกค้า "
                "สามารถสรุปข้อมูลซับซ้อนให้เป็นคำตอบสั้นๆ ที่ลูกค้าเข้าใจได้ง่าย "
                "และเหมาะสมกับการสนทนาด้วยเสียง"
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
    )
        self._cached_agents['response'] = agent
        return agent
    
    def _create_tasks(self, user_input: str, conversation_history: Optional[List[Dict]] = None) -> List[Task]:
        """Create hierarchical tasks for processing user input."""
        
        # Context from conversation history
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            history_context = "\n".join([
                f"- {turn.get('role', 'user')}: {turn.get('content', '')}"
                for turn in recent_history
            ])
        
        # Task 1: Intent routing by supervisor
        routing_task = Task(
            description=(
                f"วิเคราะห์คำถามลูกค้าและกำหนดผู้เชี่ยวชาญที่เหมาะสม:\n\n"
                f"คำถาม: '{user_input}'\n\n"
                + (f"ประวัติการสนทนา:\n{history_context}\n\n" if history_context else "")
                + "กำหนดประเภท:\n"
                "- PRODUCT: สินค้า SKU การคืนสินค้า ข้อมูลเจ้าของ\n"
                "- KNOWLEDGE: นโยบาย การรับประกัน ขั้นตอนต่างๆ\n"
                "- GENERAL: คำทักทาย ข้อมูลทั่วไป\n\n"
                "ตอบแค่ PRODUCT, KNOWLEDGE, หรือ GENERAL พร้อมเหตุผลสั้นๆ"
            ),
            agent=self._create_supervisor_agent(),
            expected_output="ประเภทคำถาม (PRODUCT/KNOWLEDGE/GENERAL) พร้อมเหตุผลสั้นๆ"
        )
        
        # Task 2: Information gathering (conditional)
        info_task = Task(
            description=(
                f"ตามคำแนะนำของ Supervisor ให้ค้นหาข้อมูลที่เกี่ยวข้องกับคำถาม: '{user_input}'\n\n"
                "หากเป็น PRODUCT: ใช้ product_database_tool หรือ google_sheets_tool\n"
                "หากเป็น KNOWLEDGE: ใช้ rag_tool\n"
                "หากเป็น GENERAL: ใช้ time_tool หรือตอบโดยตรง\n\n"
                "ค้นหาข้อมูลที่เกี่ยวข้องและสรุปผลลัพธ์อย่างกระชับ"
            ),
            agent=self._create_product_agent(),  # Will delegate based on routing
            expected_output="ข้อมูลที่ค้นหาได้พร้อมรายละเอียดสำคัญ",
            context=[routing_task]
        )
        
        # Task 3: Response composition
        response_task = Task(
            description=(
                "จากข้อมูลที่ได้รับ จัดทำคำตอบสุดท้ายที่:\n"
                "- สุภาพและเป็นกันเอง\n"
                "- กระชับ (ไม่เกิน 3-4 ประโยค)\n"
                "- ตอบตรงประเด็น\n"
                "- เหมาะสมกับการสนทนาด้วยเสียง\n"
                "- มีข้อมูลที่นำไปใช้ได้จริง\n\n"
                "หลีกเลี่ยงข้อความยาวหรือรายละเอียดเทคนิคมากเกินไป"
            ),
            agent=self._create_response_agent(),
            expected_output="คำตอบสุดท้ายสำหรับลูกค้า (กระชับและเหมาะสมกับ Voice Chat)",
            context=[routing_task, info_task]
        )
        
        return [routing_task, info_task, response_task]

    def _fast_intent_classifier(self, text: str) -> str:
        """Very small heuristic intent classifier.

        Returns one of: 'PRODUCT', 'KNOWLEDGE', 'GENERAL'.
        Designed to be extremely fast (regex/keyword-based).
        """
        t = text.lower()
        # SKU-like pattern
        if re.search(r"[a-z]{2,3}\d{3}", t) or re.search(r"รหัส\s*[:\-]?\s*[a-zA-Z0-9]{3,}", t):
            return 'PRODUCT'

        # Keywords for product queries
        product_keywords = ['สินค้า', 'รหัส', 'sku', 'model', 'ราคา', 'มีสินค้ารหัส']
        if any(k in t for k in product_keywords):
            return 'PRODUCT'

        # Knowledge / policy keywords
        knowledge_keywords = ['นโยบาย', 'รับประกัน', 'การคืน', 'เงื่อนไข', 'ขั้นตอน', 'คู่มือ']
        if any(k in t for k in knowledge_keywords):
            return 'KNOWLEDGE'

        # Greetings and general
        general_keywords = ['สวัสดี', 'ขอบคุณ', 'ราคาโดยรวม', 'ช่วย', 'สอบถาม']
        if any(k in t for k in general_keywords):
            return 'GENERAL'

        # Default to GENERAL
        return 'GENERAL'

    def _compose_final_response(self, tool_response: str, intent: str, user_input: str) -> str:
        """Create a short, polite, voice-friendly final response from a tool output.

        This is used by the fast-path to avoid running the full Response Agent while
        still returning a finalized answer to the caller.
        """
        # PRODUCT: try to find a matching product in the mock DB for a concise reply
        if intent == 'PRODUCT':
            # Try SKU match first
            sku_match = re.search(r"[A-Za-z]{2,3}\d{3}", user_input)
            sku = sku_match.group(0) if sku_match else None
            matched = None
            if sku:
                for p in MOCK_PRODUCTS:
                    if p['sku'].lower() == sku.lower():
                        matched = p
                        break

            # If no SKU, try matching by product name or owner tokens
            if not matched:
                for p in MOCK_PRODUCTS:
                    if p['product_name'].lower() in user_input.lower() or p['owner_name'].split()[0] in user_input:
                        matched = p
                        break

            if matched:
                # Compose a short, customer-facing reply in Thai
                return (
                    f"เรียนคุณ {matched['owner_name']}, สินค้า {matched['product_name']} (รหัส {matched['sku']}) "
                    f"สามารถคืนได้ที่ {matched['returned_location']} (วันที่คืน: {matched['returned_date']}) หากต้องการความช่วยเหลือเพิ่มเติม "
                    "กรุณาติดต่อสาขาหรือแจ้งข้อมูลเพิ่มเติมครับ"
                )

            # Fallback: wrap the raw tool response into a short reply
            return f"ผลลัพธ์การค้นหา: {tool_response}. หากต้องการรายละเอียดเพิ่มเติม โปรดระบุรหัสสินค้า หรือข้อมูลเจ้าของเพิ่มเติมครับ"

        # KNOWLEDGE: summarize or rephrase the RAG/tool output politely
        if intent == 'KNOWLEDGE':
            # Remove common prefixes from mock RAG outputs
            resp = tool_response
            resp = re.sub(r"^Mock RAG:\s*", "", resp)
            # Keep it short and voice-friendly
            return f"สรุปข้อมูลที่เกี่ยวข้อง: {resp} หากต้องการเอกสารหรือรายละเอียดเพิ่มเติม แจ้งได้เลยครับ"

        # GENERAL or default: return a polite canned or wrapped response
        return tool_response
    
    def process_voice_input(self, text: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process voice input through the hierarchical multi-agent system.
        
        Args:
            text: User input from speech-to-text
            conversation_history: Previous conversation context
            
        Returns:
            Dict with response, model used, and processing time
        """
        start_time = datetime.now()
        
        # Refresh config in case environment changed
        self.config.refresh()
        
        # Create and execute crew
        tasks = self._create_tasks(text, conversation_history)

        # Fast-path: if speed_mode is enabled, use an intent classifier to
        # short-circuit common query types and avoid running the full Crew.
        if getattr(self.config, 'speed_mode', False):
            intent = self._fast_intent_classifier(text)
            if intent == 'PRODUCT':
                # Let the Product Agent gather info and the Response Agent compose the final reply.
                # Build two small tasks: info gathering (product) and response composition.
                try:
                    sku_match = re.search(r"[A-Za-z]{2,3}\d{3}", text)
                    sku = sku_match.group(0) if sku_match else None

                    # Info task: ask product agent to search by SKU or by term
                    info_desc = (
                        f"ค้นหาข้อมูลที่เกี่ยวข้องกับคำถามลูกค้า: '{text}'\n"
                        + (f"หากพบ รหัสสินค้า: {sku} ให้แสดงรายละเอียดสินค้าและข้อมูลการคืน\n" if sku else "ค้นหาสินค้าจากคำอธิบายหรือชื่อเจ้าของ\n")
                        + "ให้สรุปข้อมูลสำคัญที่ตอบคำถามได้อย่างกระชับ"
                    )
                    info_task = Task(
                        description=info_desc,
                        agent=self._create_product_agent(),
                        expected_output="ข้อมูลที่ค้นหาได้พร้อมรายละเอียดสำคัญ"
                    )

                    # Response task: compose a short voice-friendly reply based on info_task
                    response_task = Task(
                        description=(
                            "จากข้อมูลที่ได้ สร้างคำตอบสั้น กระชับ และสุภาพสำหรับลูกค้าผ่านระบบเสียง (ไม่เกิน 3 ประโยค)"
                        ),
                        agent=self._create_response_agent(),
                        expected_output="คำตอบสุดท้ายสำหรับลูกค้า (กระชับและเหมาะสมกับ Voice Chat)",
                        context=[info_task]
                    )

                    small_crew = Crew(
                        agents=[self._create_product_agent(), self._create_response_agent()],
                        tasks=[info_task, response_task],
                        process=Process.hierarchical,
                        manager_llm=self.llm,
                        verbose=False,
                        max_rpm=200,
                    )

                    result = small_crew.kickoff()
                    response_text = str(result).strip()
                    # Clean and normalize output similar to main flow
                    if "Final Answer:" in response_text:
                        response_text = response_text.split("Final Answer:")[-1].strip()
                    response_text = response_text.replace('\n', ' ')
                    response_text = response_text.replace('**', '')
                    response_text = re.sub(r'[^a-zA-Z0-9\s\u0E00-\u0E7F]', '', response_text)
                    response_text = ' '.join(response_text.split())

                    processing_time = (datetime.now() - start_time).total_seconds()
                    return {
                        'response': response_text,
                        'model': self.config.model,
                        'processing_time_seconds': processing_time,
                        'tools_available': list(self.tools.keys()),
                        'timestamp': datetime.now().isoformat(),
                        'fast_path': True,
                        'agent_path': 'product+response'
                    }
                except Exception as _e:
                    # Fallback to fast composer if the small crew fails
                    logger.warning(f"Product fast-crew failed, falling back to composer: {_e}")
                    if sku:
                        resp = self.tools['products']._run(query_type='get_by_sku', sku=sku)
                    else:
                        resp = self.tools['products']._run(query_type='search', search_term=text)
                    final_resp = self._compose_final_response(resp, intent='PRODUCT', user_input=text)
                    processing_time = (datetime.now() - start_time).total_seconds()
                    return {
                        'response': final_resp,
                        'model': self.config.model,
                        'processing_time_seconds': processing_time,
                        'tools_available': list(self.tools.keys()),
                        'timestamp': datetime.now().isoformat(),
                        'fast_path': True,
                        'fallback': True,
                    }

            if intent == 'KNOWLEDGE':
                try:
                    # Info task: ask knowledge agent to search documents
                    info_task = Task(
                        description=(
                            f"ค้นหาเอกสารหรือข้อกำหนดที่เกี่ยวข้องกับคำถาม: '{text}'\n"
                            "ให้สรุปประเด็นสำคัญที่เกี่ยวข้องและหลักฐานอ้างอิงสั้นๆ"
                        ),
                        agent=self._create_knowledge_agent(),
                        expected_output="ข้อความสรุปจากเอกสารที่เกี่ยวข้อง",
                    )

                    response_task = Task(
                        description=(
                            "จากข้อมูลที่ได้ สร้างคำตอบสั้น กระชับ และสุภาพสำหรับลูกค้าผ่านระบบเสียง (ไม่เกิน 3 ประโยค)"
                        ),
                        agent=self._create_response_agent(),
                        expected_output="คำตอบสุดท้ายสำหรับลูกค้า (กระชับและเหมาะสมกับ Voice Chat)",
                        context=[info_task]
                    )

                    small_crew = Crew(
                        agents=[self._create_knowledge_agent(), self._create_response_agent()],
                        tasks=[info_task, response_task],
                        process=Process.hierarchical,
                        manager_llm=self.llm,
                        verbose=False,
                        max_rpm=200,
                    )

                    result = small_crew.kickoff()
                    response_text = str(result).strip()
                    if "Final Answer:" in response_text:
                        response_text = response_text.split("Final Answer:")[-1].strip()
                    response_text = response_text.replace('\n', ' ')
                    response_text = response_text.replace('**', '')
                    response_text = re.sub(r'[^a-zA-Z0-9\s\u0E00-\u0E7F]', '', response_text)
                    response_text = ' '.join(response_text.split())

                    processing_time = (datetime.now() - start_time).total_seconds()
                    return {
                        'response': response_text,
                        'model': self.config.model,
                        'processing_time_seconds': processing_time,
                        'tools_available': list(self.tools.keys()),
                        'timestamp': datetime.now().isoformat(),
                        'fast_path': True,
                        'agent_path': 'knowledge+response'
                    }
                except Exception as _e:
                    logger.warning(f"Knowledge fast-crew failed, falling back to composer: {_e}")
                    resp = self.tools['rag']._run(query=text, top_k=2)
                    final_resp = self._compose_final_response(resp, intent='KNOWLEDGE', user_input=text)
                    processing_time = (datetime.now() - start_time).total_seconds()
                    return {
                        'response': final_resp,
                        'model': self.config.model,
                        'processing_time_seconds': processing_time,
                        'tools_available': list(self.tools.keys()),
                        'timestamp': datetime.now().isoformat(),
                        'fast_path': True,
                        'fallback': True,
                    }

            # GENERAL delegates to a short canned reply for speed
            if intent == 'GENERAL':
                resp = "สวัสดีครับ ผมช่วยอะไรได้บ้างครับ?"
                final_resp = self._compose_final_response(resp, intent='GENERAL', user_input=text)
                processing_time = (datetime.now() - start_time).total_seconds()
                return {
                    'response': final_resp,
                    'model': self.config.model,
                    'processing_time_seconds': processing_time,
                    'tools_available': list(self.tools.keys()),
                    'timestamp': datetime.now().isoformat(),
                    'fast_path': True,
                }
        
        # Use hierarchical process for speed
        crew = Crew(
            agents=[self._create_supervisor_agent(), 
                   self._create_product_agent(),
                   self._create_knowledge_agent(), 
                   self._create_response_agent()],
            tasks=tasks,
            process=Process.hierarchical,  # Hierarchical for speed
            manager_llm=self.llm,
            verbose=True,
            max_rpm=100,  # Increased rate limit for speed
        )
        
        try:
            result = crew.kickoff()
            response_text = str(result).strip()
            
            # Clean up response formatting
            if "Final Answer:" in response_text:
                response_text = response_text.split("Final Answer:")[-1].strip()
            
            # Text preprocessing: remove unwanted symbols
            response_text = response_text.replace('\n', ' ')  # Remove newlines
            response_text = response_text.replace('**', '')  # Remove asterisks
            response_text = re.sub(r'[^a-zA-Z0-9\s\u0E00-\u0E7F]', '', response_text)  # Keep English letters, numbers, spaces, and Thai characters
            response_text = ' '.join(response_text.split())  # Normalize spaces
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": response_text,
                "model": self.config.model,
                "processing_time_seconds": processing_time,
                "tools_available": list(self.tools.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # If the exception indicates a missing coworker/context validation in
            # AskQuestionToolSchema, short-circuit to a fast heuristic responder
            msg = str(e)
            if 'AskQuestionToolSchema' in msg or 'coworker' in msg or 'context' in msg:
                logger.warning(f"AskQuestion tool validation issue detected, using fast fallback: {msg}")
                # Fast heuristic: check for SKU-like tokens, refund keywords, or default
                text_lower = text.lower()
                sku_match = re.search(r"[A-Za-z]{2,3}\d{3}", text)
                if sku_match:
                    sku = sku_match.group(0)
                    resp = self.tools['products']._run(query_type='get_by_sku', sku=sku)
                elif any(k in text_lower for k in ['คืนสินค้า', 'การคืน', 'คืนเงิน', 'return']):
                    resp = self.tools['rag']._run(query='นโยบายการคืนสินค้า', top_k=2)
                else:
                    # Generic short reply using time_tool or canned response
                    resp = "ขอทราบรายละเอียดเพิ่มเติมเล็กน้อยได้ไหมครับ เช่น รหัสสินค้า หรือสาขาที่ต้องการคืนสินค้า"

                # Compose a final customer-friendly response for the fallback
                intent_guess = 'PRODUCT' if sku_match else ('KNOWLEDGE' if any(k in text_lower for k in ['คืนสินค้า', 'การคืน', 'คืนเงิน', 'return']) else 'GENERAL')
                final_resp = self._compose_final_response(resp, intent=intent_guess, user_input=text)

                processing_time = (datetime.now() - start_time).total_seconds()
                return {
                    "response": final_resp,
                    "model": self.config.model,
                    "processing_time_seconds": processing_time,
                    "tools_available": list(self.tools.keys()),
                    "timestamp": datetime.now().isoformat(),
                    "fallback": True,
                }

            logger.error(f"Error processing voice input: {e}")
            return {
                "response": f"ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล: {str(e)}",
                "model": self.config.model,
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for health checks."""
        return {
            "engine": "crewai_hierarchical",
            "model": self.config.model,
            "base_url": self.config.base_url,
            "tools": {
                "time": True,
                "product_database": True,
                "google_sheets": SHEETS_AVAILABLE,
                "rag_pdf": RAG_AVAILABLE
            },
            "mock_products_count": len(MOCK_PRODUCTS),
            "ready": True,
            "architecture": "supervisor -> specialists -> response"
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    system = VoiceCallCenterMultiAgent()
    
    # Test queries
    test_queries = [
        "สวัสดีครับ ขอสอบถามแพ็กเกจอินเทอร์เน็ตหน่อย",
        "มีสินค้ารหัส TEL001 อะไรบ้างครับ",
        "นโยบายการคืนสินค้าเป็นยังไงบ้างครับ",
        "ขอดูข้อมูลของนายสมชาย ใจดี หน่อยครับ"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"ลูกค้า: {query}")
        result = system.process_voice_input(query)
        print(f"ระบบ: {result['response']}")
        print(f"เวลาประมวลผล: {result.get('processing_time_seconds', 0):.2f} วินาที")
    
    # System status
    print(f"\n{'='*50}")
    print("สถานะระบบ:")
    status = system.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
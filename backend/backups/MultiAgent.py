"""MultiAgent orchestration built with CrewAI (LiteLLM backend).

Provides a tiny two-agent pipeline (intent analysis + response composition)
for Thai call-center style responses.

Environment variables (evaluated at runtime):
    TOGETHER_API_KEY or OPENROUTER_API_KEY (required)
    TOGETHER_BASE_URL or OPENROUTER_BASE_URL (default: auto-detected)
    LLM_MODEL (default: together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo)
    
Note: Will automatically use OpenRouter if OPENROUTER_API_KEY is found,
      or Together AI if TOGETHER_API_KEY is found.

Usage:
    from MultiAgent import MultiAgent
    ma = MultiAgent()
    result = ma.run("สวัสดีครับ")
    print(result["response"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

try:
    from crewai import Agent, Task, Crew, Process
    try:
        from crewai import LLM  # Newer versions
    except Exception:
        from crewai.llm import LLM  # Older path
except Exception as e:  # pragma: no cover
    raise ImportError("crewai is required. Install with: pip install crewai litellm") from e


logger = logging.getLogger(__name__)

TOGETHER_MODEL = "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"
OPENROUTER_MODEL = "openrouter/qwen/qwen3-4b:free"


def _late_env_hydrate():
    """Attempt late .env loading by traversing parent directories until found."""
    if os.getenv("TOGETHER_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
        return
    
    tried: List[str] = []
    current = os.path.dirname(__file__)
    
    # Also check the parent directory (root of project)
    search_paths = [
        current,  # backend directory
        os.path.dirname(current),  # parent directory (project root)
    ]
    
    for base_path in search_paths:
        for _ in range(6):  # traverse up to 6 levels
            env_path = os.path.join(base_path, '.env')
            tried.append(env_path)
            if os.path.exists(env_path):
                try:
                    logger.debug(f"Found .env file at: {env_path}")
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if not s or s.startswith('#') or '=' not in s:
                                continue
                            k, v = s.split('=', 1)
                            k = k.strip(); v = v.strip().strip('"').strip("'")
                            if k and v and k not in os.environ:
                                os.environ[k] = v
                                logger.debug(f"Loaded env var: {k}")
                    if os.getenv("TOGETHER_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
                        logger.debug(f"Successfully loaded API key from {env_path}")
                        return
                except Exception as e:
                    logger.debug(f"Error reading {env_path}: {e}")
            parent = os.path.dirname(base_path)
            if parent == base_path:
                break
            base_path = parent
    
    logger.debug(f"Env key not found; searched: {tried}")


@dataclass
class MultiAgentConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", TOGETHER_MODEL))
    temperature: float = 0.3
    max_tokens: int = 1024
    api_key: Optional[str] = None  # resolved later
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("TOGETHER_BASE_URL") or "https://api.together.xyz/v1")
    request_timeout: int = 60

    def resolve(self):
        # Try OpenRouter first, then Together AI
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if self.api_key:
                # Update base URL for OpenRouter
                self.base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
                # Update model for OpenRouter if using default - keep provider prefix for LiteLLM
                if self.model == TOGETHER_MODEL:
                    self.model = OPENROUTER_MODEL  # Full provider prefix
        
        if not self.api_key:
            self.api_key = os.getenv("TOGETHER_API_KEY")
            if self.api_key:
                self.base_url = os.getenv("TOGETHER_BASE_URL") or "https://api.together.xyz/v1"
                # Keep original model for Together AI (already has together_ai/ prefix)
        
        if not self.api_key:
            _late_env_hydrate()
            # Try again after loading .env
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if self.api_key:
                self.base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
                if self.model == TOGETHER_MODEL:
                    self.model = OPENROUTER_MODEL
            else:
                self.api_key = os.getenv("TOGETHER_API_KEY")
                if self.api_key:
                    self.base_url = os.getenv("TOGETHER_BASE_URL") or "https://api.together.xyz/v1"
        
        return self

    def refresh(self):
        """Re-read environment (useful in dynamic notebooks like Colab after setting %env)."""
        old_key = self.api_key
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("TOGETHER_API_KEY") or self.api_key
        
        # Update base URL if API key source changed
        if self.api_key != old_key:
            if os.getenv("OPENROUTER_API_KEY"):
                self.base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
                if self.model == TOGETHER_MODEL:
                    self.model = OPENROUTER_MODEL
            elif os.getenv("TOGETHER_API_KEY"):
                self.base_url = os.getenv("TOGETHER_BASE_URL") or "https://api.together.xyz/v1"
        
        return self


class MultiAgent:
    """A tiny CrewAI orchestrator to generate responses from multiple agents.

    Usage:
        ma = MultiAgent()
        result = ma.run("สวัสดีครับ ขอสอบถามแพ็กเกจอินเทอร์เน็ตหน่อย")
        print(result["response"])  # str
    """

    def __init__(self, config: Optional[MultiAgentConfig] = None) -> None:
        # Ensure .env is loaded before anything else
        _late_env_hydrate()
        
        self.config = (config or MultiAgentConfig()).resolve()
        if not self.config.api_key:
            raise RuntimeError("Missing TOGETHER_API_KEY or OPENROUTER_API_KEY. Set one of them before use.")

        # Determine which provider we're using
        self.provider = "openrouter" if "openrouter.ai" in self.config.base_url else "together"
        
        logger.info(
            "MultiAgent init | provider=%s | model=%s | base_url=%s | key_prefix=%s",
            self.provider,
            self.config.model,
            self.config.base_url,
            self.config.api_key[:8] + "…",
        )

        # Initialize CrewAI's native LLM (LiteLLM backend) - exactly like the working notebook
        # Ensure the API key is set in environment for LiteLLM
        if self.config.api_key:
            if self.provider == "openrouter":
                os.environ["OPENROUTER_API_KEY"] = self.config.api_key
            else:
                os.environ["TOGETHER_API_KEY"] = self.config.api_key
        
        try:
            # Configure LLM - the model name already includes the provider prefix for LiteLLM
            self.llm = LLM(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed initializing LLM. Model={self.config.model} base_url={self.config.base_url} msg={e}. "
                "Ensure crewai & litellm are up to date and that TOGETHER_API_KEY is valid."
            ) from e

        # Define agents
        self.intent_analyst = Agent(
            role="Thai Intent Analyst",
            goal=(
                "วิเคราะห์เจตนา (intent) และข้อมูลสำคัญจากข้อความของผู้ใช้ภาษาไทย "
                "สรุปประเด็น คำสำคัญ และบริบทที่เกี่ยวข้องอย่างกระชับ"
            ),
            backstory=(
                "คุณเป็นผู้เชี่ยวชาญด้านภาษาไทยในงาน Call Center "
                "สามารถจับประเด็นและตีความความต้องการของลูกค้าได้อย่างแม่นยำ"
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
        )

        self.response_composer = Agent(
            role="Thai Response Composer",
            goal=(
                "จัดทำคำตอบภาษาไทยที่สุภาพ ชัดเจน และนำไปใช้ได้จริง ตามนโยบายทั่วไปของฝ่ายบริการลูกค้า"
            ),
            backstory=(
                "คุณเป็นผู้เชี่ยวชาญการสื่อสารเชิงบริการลูกค้า "
                "ให้ข้อมูลและแนวทางแก้ไขอย่างเป็นขั้นตอน พร้อมสรุปสั้นท้ายข้อความ"
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
        )

    def _build_crew(self, user_text: str, conversation_history: Optional[List[Dict[str, Any]]]) -> Crew:
        history_text = ""
        if conversation_history:
            # Flatten brief history for context
            pairs = []
            for turn in conversation_history[-6:]:  # last 6 turns
                role = turn.get("role", "user")
                content = turn.get("content", "")
                pairs.append(f"- {role}: {content}")
            history_text = "\n".join(pairs)

        analyze_task = Task(
            description=(
                "วิเคราะห์ข้อความของผู้ใช้และระบุ: เจตนา, ประเด็นหลัก, คำสำคัญ, "
                "ข้อมูลที่ขาดหาย, และความเร่งด่วน (ถ้ามี).\n\n"
                f"ข้อความผู้ใช้: '''{user_text}'''\n\n"
                + (f"ประวัติสนทนาล่าสุด:\n{history_text}\n\n" if history_text else "")
                + "ให้ผลลัพธ์เป็น bullet list ภาษาไทย"
            ),
            agent=self.intent_analyst,
            expected_output=(
                "Bullet list สั้นๆ ที่สรุปเจตนา ประเด็น คีย์เวิร์ด ข้อมูลที่ขาด และความเร่งด่วน"
            ),
        )

        compose_task = Task(
            description=(
                "จากผลการวิเคราะห์ สร้างคำตอบภาษาไทยที่:\n"
                "- สุภาพ ชัดเจน กระชับ สั้นๆ เหมาะสมกับลูกค้าไทย\n"
                "- ให้ทางเลือกหรือขั้นตอนถัดไปที่ทำได้ทันที\n"
                "- ถ้ามีข้อจำกัด ให้แจ้งอย่างโปร่งใส\n"
            ),
            agent=self.response_composer,
            expected_output="คำตอบสุดท้ายภาษาไทยที่พร้อมส่งให้ลูกค้า",
            context=[analyze_task],
        )

        return Crew(
            agents=[self.intent_analyst, self.response_composer],
            tasks=[analyze_task, compose_task],
            process=Process.sequential,
            verbose=False,
        )

    def run(self, text: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run the crew with the given user text and optional history.

        Returns a dict with keys: response, model, used_base_url
        """
        # Refresh key in case user set %env after object creation (e.g., in Colab)
        self.config.refresh()
        if not self.config.api_key:
            raise RuntimeError("API key missing at runtime. Set TOGETHER_API_KEY or OPENROUTER_API_KEY before calling run().")
        
        # Ensure the environment variable is updated for LiteLLM
        if self.provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = self.config.api_key
        else:
            os.environ["TOGETHER_API_KEY"] = self.config.api_key
        
        crew = self._build_crew(text, conversation_history)
        output = crew.kickoff()
        # CrewAI returns a result object or str depending on version; coerce to str
        final_text = str(output)
        return {
            "response": final_text.strip(),
            "model": self.config.model,
            "used_base_url": self.config.base_url,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return LLM orchestration status for health checks."""
        return {
            "engine": "crewai",
            "model": self.config.model,
            "base_url": self.config.base_url,
            "ready": True,
        }
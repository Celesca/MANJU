"""
MultiAgent orchestration built with CrewAI.

This module exposes a simple MultiAgent class that coordinates a few
lightweight agents to analyze user input and compose a helpful reply.

Environment variables used:
- OPENAI_API_KEY or OPENROUTER_API_KEY
- OPENAI_BASE_URL or OPENROUTER_BASE_URL (optional, for OpenRouter or custom gateways)
- LLM_MODEL (optional; default: gpt-4o-mini)

Dependencies:
- crewai
- langchain-openai
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    # Core CrewAI constructs
    from crewai import Agent, Task, Crew, Process
except Exception as e:  # pragma: no cover - helpful error if not installed
    raise ImportError(
        "crewai is required. Please install dependencies (e.g. `pip install crewai langchain langchain-openai`)."
    ) from e

try:
    # LangChain OpenAI chat model wrapper
    from langchain_openai import ChatOpenAI
except Exception as e:  # pragma: no cover
    raise ImportError(
        "langchain-openai is required. Install via `pip install langchain-openai`."
    ) from e


@dataclass
class MultiAgentConfig:
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature: float = 0.3
    max_tokens: int = 1024
    # Support custom base URL (e.g., OpenRouter) and API keys
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url: Optional[str] = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    request_timeout: int = 60


class MultiAgent:
    """A tiny CrewAI orchestrator to generate responses from multiple agents.

    Usage:
        ma = MultiAgent()
        result = ma.run("สวัสดีครับ ขอสอบถามแพ็กเกจอินเทอร์เน็ตหน่อย")
        print(result["response"])  # str
    """

    def __init__(self, config: Optional[MultiAgentConfig] = None) -> None:
        self.config = config or MultiAgentConfig()

        if not self.config.api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY in environment."
            )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

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
                "- สุภาพ ชัดเจน เหมาะสมกับลูกค้าไทย\n"
                "- ให้ทางเลือกหรือขั้นตอนถัดไปที่ทำได้ทันที\n"
                "- ถ้ามีข้อจำกัด ให้แจ้งอย่างโปร่งใส\n"
                "- ลงท้ายด้วยสรุป 1 บรรทัด\n"
                "ระยะยาวไม่เกิน 8-12 บรรทัด\n"
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
        crew = self._build_crew(text, conversation_history)
        output = crew.kickoff()
        # CrewAI returns a result object or str depending on version; coerce to str
        final_text = str(output)
        return {
            "response": final_text.strip(),
            "model": self.config.model,
            "used_base_url": self.config.base_url,
        }

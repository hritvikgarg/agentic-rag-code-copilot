"""Architecture guards for the external-LLM boundary."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "copilot"
SDK_IMPORT = re.compile(r"^\s*(import|from)\s+google(\.genai|\s+import\s+genai)\b", re.MULTILINE)


def test_only_the_gemini_module_imports_the_provider_sdk():
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if SDK_IMPORT.search(path.read_text()) and path.name != "gemini.py"
    ]
    assert offenders == []


def test_the_gemini_module_has_no_retrieval_or_rag_imports():
    text = (SRC / "llm" / "gemini.py").read_text()
    assert "copilot.retrieval" not in text and "copilot.rag" not in text


def test_the_services_and_factory_route_every_call_through_the_guard():
    service = (SRC / "rag" / "service.py").read_text()
    assert service.count("GuardedLLMClient(") == 2  # answer_plain and RagService.__init__
    assert "self._llm.generate" in service and "_inner" not in service
    factory = (SRC / "llm" / "factory.py").read_text()
    assert "GuardedLLMClient(" in factory


def test_only_the_protocol_the_guard_gemini_and_fake_define_generate():
    implementers = sorted(
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if re.search(r"^\s+def generate\(self", p.read_text(), re.MULTILINE)
    )
    assert implementers == ["llm/base.py", "llm/fake.py", "llm/gemini.py", "llm/guarded.py"]

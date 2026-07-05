"""
tests/test_model_factory.py
멀티 모델 추상화 팩토리 테스트.

[검증]
1. provider × role → 모델명 매핑 (기본값)
2. .env 오버라이드가 기본값보다 우선
3. 알 수 없는 provider → openai 폴백
4. provider 전환 시 llm 캐시가 새 provider 로 갱신
5. 키 부재 시 명확한 RuntimeError
6. 미설치 패키지(gemini/anthropic) 선택 시 안내 RuntimeError (lazy import)
"""
import pytest


def test_default_model_mapping(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL_MAIN", raising=False)
    from graph.model_factory import resolve_model_name, ModelRole
    assert resolve_model_name("openai", ModelRole.MAIN) == "gpt-5.4"
    assert resolve_model_name("openai", ModelRole.INTENT) == "gpt-5.4-mini"
    # gemini-2.5-flash → gemini-3.1-flash-lite 교체(가성비 포지션 대표)
    assert resolve_model_name("gemini", ModelRole.MAIN) == "gemini-3.1-flash-lite"
    assert resolve_model_name("anthropic", ModelRole.MAIN) == "claude-sonnet-4-6"
    assert resolve_model_name("anthropic", ModelRole.INTENT) == "claude-haiku-4-5-20251001"
    # deepseek 신규
    assert resolve_model_name("deepseek", ModelRole.MAIN) == "deepseek-v4-flash"
    assert resolve_model_name("deepseek", ModelRole.INTENT) == "deepseek-v4-flash"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL_MAIN", "gpt-5.5")
    from graph.model_factory import resolve_model_name, ModelRole
    assert resolve_model_name("openai", ModelRole.MAIN) == "gpt-5.5"


def test_unknown_provider_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "llama-local")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    from graph.model_factory import create_chat_model, ModelRole
    llm = create_chat_model(ModelRole.MAIN, temperature=0.5)
    # openai 로 폴백 → ChatOpenAI 인스턴스
    name = llm.model_name if hasattr(llm, "model_name") else None
    assert name == "gpt-5.4"


def test_provider_switch_refreshes_cache(monkeypatch):
    """LLM_PROVIDER 변경 시 (provider,role) 키가 달라져 새 인스턴스를 생성한다."""
    import graph.llm as llm_mod
    # 캐시 초기화
    llm_mod._llm_cache.clear()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    m1 = llm_mod.get_main_llm(0.5)
    name1 = m1.bound.model_name if hasattr(m1, "bound") else m1.model_name
    assert name1 == "gpt-5.4"

    # provider 를 anthropic 으로 전환 → 모델명이 바뀌어야 함
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    # anthropic 패키지가 없으면 RuntimeError(안내) → 캐시 키 전환 자체는 검증됨
    try:
        m2 = llm_mod.get_main_llm(0.5)
        inner = m2.bound if hasattr(m2, "bound") else m2
        # provider 마다 모델명 속성이 다름: ChatOpenAI/Gemini=model_name, ChatAnthropic=model
        name2 = getattr(inner, "model_name", None) or getattr(inner, "model", None)
        assert name2 == "claude-sonnet-4-6"
    except RuntimeError as e:
        assert "langchain-anthropic" in str(e)


def test_missing_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import graph.llm as llm_mod
    llm_mod._llm_cache.clear()
    from graph.model_factory import create_chat_model, ModelRole
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        create_chat_model(ModelRole.MAIN)


def test_gemini_lazy_import_message(monkeypatch):
    """gemini 패키지 미설치 시 안내 메시지 포함 RuntimeError (lazy import)."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    from graph.model_factory import create_chat_model, ModelRole
    try:
        import langchain_google_genai  # noqa
        pytest.skip("langchain-google-genai 가 설치되어 있어 lazy import 실패를 검증 불가")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="langchain-google-genai"):
        create_chat_model(ModelRole.MAIN)

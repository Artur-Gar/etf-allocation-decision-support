"""Test live OpenAI ETF classification with minimal contract-style assertions.

The scope covers prompt rendering, API request wiring, response parsing, and
basic semantic quality for a couple of clear ETF examples. These tests do not
try to lock down every exact LLM output because live model behavior can vary.
"""

from __future__ import annotations

import pytest
from openai import APIConnectionError, APITimeoutError

from llm.client import (
    PRIMARY_REGION_VALUES,
    SECTOR_FOCUS_VALUES,
    STYLE_FOCUS_VALUES,
    classify_prompt,
    get_client,
    get_model_name,
    has_live_openai_config,
    load_env_file,
)
from llm.workflow import DEFAULT_TEMPLATE_PATH, load_template, render_prompt


load_env_file()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not has_live_openai_config(),
        reason="OPENAI_API_KEY and OPENAI_MODEL must be set to run live integration tests.",
    ),
]


@pytest.fixture(scope="module")
def live_results() -> dict[str, object]:
    """Classify two obvious ETFs once for the full module."""
    template = load_template(DEFAULT_TEMPLATE_PATH)
    client = get_client()
    results: dict[str, object] = {}

    test_rows = [
        {
            "ticker": "IVW",
            "etf_name": "iShares S&P 500 Growth ETF",
            "provider": "iShares / BlackRock",
            "description": (
                "The fund seeks to track the investment results of an index composed "
                "of large-capitalization U.S. equities that exhibit growth characteristics."
            ),
        },
        {
            "ticker": "AAXJ",
            "etf_name": "iShares MSCI All Country Asia ex Japan ETF",
            "provider": "iShares / BlackRock",
            "description": (
                "The fund seeks to track the investment results of an index composed "
                "of large- and mid-capitalization Asian equities, excluding Japan."
            ),
        },
    ]

    try:
        for row in test_rows:
            prompt = render_prompt(template, row)
            results[row["ticker"]] = classify_prompt(
                client=client,
                prompt=prompt,
                model=get_model_name(),
            )
    except (APIConnectionError, APITimeoutError) as exc:
        pytest.skip(f"Live OpenAI classification is unreachable from this environment: {exc}")

    return results


def test_live_classification_response_shape(live_results: dict[str, object]) -> None:
    """Check that live responses land in the expected schema and value ranges."""
    for result in live_results.values():
        assert result.primary_region in PRIMARY_REGION_VALUES
        assert result.style_focus in STYLE_FOCUS_VALUES
        assert result.sector_focus in SECTOR_FOCUS_VALUES
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning_short, str)
        assert result.reasoning_short.strip()


def test_live_ivw_is_classified_as_us_growth(live_results: dict[str, object]) -> None:
    """Check that a very clear U.S. growth ETF is classified as expected."""
    ivw = live_results["IVW"]

    assert ivw.primary_region == "US"
    assert ivw.developed_or_emerging == "Developed"
    assert ivw.style_focus == "Growth"
    assert ivw.sector_focus == "Broad Market"


def test_live_aaxj_is_classified_as_broad_asia_ex_japan(live_results: dict[str, object]) -> None:
    """Check that a broad Asia ex-Japan equity ETF is classified conservatively."""
    aaxj = live_results["AAXJ"]

    assert aaxj.primary_region == "Asia Pacific ex-Japan"
    assert aaxj.style_focus == "Core"
    assert aaxj.sector_focus == "Broad Market"


from __future__ import annotations

import re


def _fields(text: str) -> list[str]:
    """Split compact pipe-delimited column definitions."""
    return [value.strip() for value in text.split("|")]


def _rule(label: str, *patterns: str) -> tuple[str, list[str]]:
    """Build one regex classification rule."""
    return label, list(patterns)


PRODUCT_PATH_PATTERN = re.compile(r"/us/products/(?P<product_id>\d+)/(?P<slug>[^/?#]+)")
LIST_COLUMNS = _fields(
    "ticker|etf_name|trailing_yield_12m_pct|trailing_yield_as_of|ytd_return_pct|"
    "ytd_return_as_of|inception_date|gross_expense_ratio_pct|net_expense_ratio_pct|net_assets_usd_list"
)
FINAL_COLUMNS = _fields(
    "ticker|etf_name|description|provider|asset_class|geographic_scope|primary_region|"
    "developed_or_emerging|style_focus|size_focus|sector_focus|active_vs_index|benchmark_index|"
    "expense_ratio|net_assets_usd|bid_ask_spread_30d|number_of_holdings|pe_ratio|pb_ratio|"
    "standard_deviation_3y|equity_beta_3y|top_country_1|top_country_1_weight|top_country_2|"
    "top_country_2_weight|top_country_3|top_country_3_weight|other_countries|other_countries_weight|"
    "top_industry_1|top_industry_1_weight|top_industry_2|top_industry_2_weight|"
    "top_industry_3|top_industry_3_weight|other_industries|other_industries_weight"
)

COMPONENT_MARKERS = {
    "keyFundFacts": '"componentId":"keyFundFacts"',
    "fundamentalsAndRisk": '"componentId":"fundamentalsAndRisk"',
    "fundHeader": '"componentId":"fundHeader"',
    "exposureBreakdowns": '"componentId":"exposureBreakdowns"',
}
COUNTRY_GROUP_KEYWORDS = [r"\bBRIC\b", r"\bG7\b", r"\bG20\b", r"\bASEAN\b", r"\bGCC\b", r"\bNordic\b"]
DEVELOPED_COUNTRIES = set(_fields(
    "Australia|Austria|Belgium|Canada|Denmark|Finland|France|Germany|Hong Kong|Ireland|Israel|"
    "Italy|Japan|Korea (South)|Netherlands|New Zealand|Norway|Portugal|Singapore|Spain|Sweden|"
    "Switzerland|Taiwan|United Kingdom|United States|US"
))
EMERGING_COUNTRIES = set(_fields(
    "Brazil|Chile|China|Colombia|Czech Republic|Egypt|Greece|Hungary|India|Indonesia|Malaysia|"
    "Mexico|Peru|Philippines|Poland|Qatar|Saudi Arabia|South Africa|Thailand|Turkey|United Arab Emirates"
))

COUNTRY_PRIMARY_REGION_RULES = [
    _rule("United Kingdom", r"\bunited kingdom\b", r"\buk\b", r"\bbritain\b"),
    _rule("South Korea", r"\bsouth korea\b", r"\bkorea\b"),
    *[_rule(country, rf"\b{country.lower()}\b") for country in _fields(
        "Hong Kong|Taiwan|Singapore|Australia|Canada|Mexico|Brazil|South Africa|Saudi Arabia|Kuwait|"
        "Qatar|Turkey|Chile|Peru|Colombia|Indonesia|Malaysia|Philippines|Thailand|Poland|Israel|"
        "Denmark|Finland|Norway|Sweden|Germany|France|Italy|Spain|Belgium|Austria|Switzerland|"
        "Netherlands|New Zealand"
    )],
]
SINGLE_COUNTRY_PRIMARY_REGIONS = {"US", "China", "India", "Japan", *{label for label, _ in COUNTRY_PRIMARY_REGION_RULES}}
PRIMARY_REGION_RULES = [
    _rule("Global ex-US", r"\bglobal ex[- ]u\.s\.\b", r"\bacwi ex[- ]u\.s\.\b", r"\ball country world ex[- ]u\.s\.\b", r"\bworld ex[- ]u\.s\.\b"),
    _rule("Global", r"\bacwi\b", r"\ball country world\b", r"\bglobal\b", r"\bworld\b"),
    *COUNTRY_PRIMARY_REGION_RULES,
    _rule("Asia Pacific ex-Japan", r"\basia ex japan\b", r"\basia/pacific\b", r"\basia 50\b", r"\bpacific ex[- ]japan\b"),
    _rule("Emerging Markets", r"\bemerging markets\b", r"\bmsci em\b"),
    _rule("Developed ex-US", r"\bdeveloped ex[- ]u\.s\.\b", r"\bdeveloped markets ex[- ]u\.s\.\b", r"\beafe\b"),
    _rule("China", r"\bchina\b"),
    _rule("India", r"\bindia\b"),
    _rule("Japan", r"\bjapan\b"),
    _rule("Latin America", r"\blatin america\b"),
    _rule("Europe", r"\beurope\b", r"\beuro\b", r"\beurozone\b"),
    _rule("US", r"u\.s\.", r"\busa\b", r"\bs&p 500\b", r"\brussell\b", r"dow jones u\.s\.", r"\btreasury\b", r"\bmunicipal\b", r"\bmortgage\b"),
]
STYLE_RULES = [
    _rule("ESG", r"\besg\b", r"\baware\b", r"\bscreened\b", r"\bsustainable\b"),
    _rule("Momentum", r"\bmomentum\b"),
    _rule("Quality", r"\bquality\b"),
    _rule("Growth", r"\bgrowth\b"),
    _rule("Value", r"\bvalue\b"),
    _rule("Low Volatility", r"\bminimum volatility\b", r"\bmin vol\b", r"\blow volatility\b"),
    _rule("Multi-Factor", r"\bmultifactor\b", r"\bmulti[- ]factor\b"),
    _rule("Dividend", r"\bdividend\b", r"\bbuywrite\b", r"\bpremium income\b", r"\bincome\b", r"\byield\b"),
    _rule("Thematic", r"\bbitcoin\b", r"\blithium\b", r"\bmetals\b", r"\bminers\b", r"\bpower infrastructure\b", r"\bclean energy\b", r"\brobotics\b", r"\bcyber\b", r"\bblockchain\b", r"\bdigital\b"),
]
SIZE_RULES = [
    _rule("Large/Mid Cap", r"\blarge[- ]and[- ]mid[- ]cap\b", r"\blarge- and mid-cap\b", r"\blarge and mid cap\b", r"\blarge[- ]mid cap\b"),
    _rule("Large Cap", r"\blarge[- ]cap\b", r"\bs&p 500\b", r"\brussell 1000\b"),
    _rule("Mid Cap", r"\bmid[- ]cap\b", r"\brussell midcap\b", r"\bs&p midcap\b"),
    _rule("Small Cap", r"\bsmall[- ]cap\b", r"\brussell 2000\b", r"\bs&p smallcap\b", r"\bmicro[- ]cap\b"),
]
SECTOR_RULES = [
    _rule("Real Estate", r"\breal estate\b", r"\breit\b"),
    _rule("Technology", r"\btechnology\b", r"\btech\b", r"\bsemiconductor\b"),
    _rule("Healthcare", r"\bhealth care\b", r"\bhealthcare\b", r"\bmedical\b", r"\bbiotech\b"),
    _rule("Financials", r"\bfinancials\b", r"\bbank\b", r"\bbanks\b", r"\binsurance\b"),
    _rule("Energy", r"\benergy\b", r"\boil\b", r"\bgas\b"),
    _rule("Industrials", r"\bindustrials\b", r"\binfrastructure\b", r"\btransport\b", r"\baerospace\b"),
    _rule("Consumer", r"\bconsumer\b", r"\bretail\b", r"\bdiscretionary\b", r"\bstaples\b"),
    _rule("Utilities", r"\butilities\b"),
    _rule("Thematic", r"\bbitcoin\b", r"\blithium\b", r"\bmetals\b", r"\bthematic\b", r"\bclean energy\b", r"\brobotics\b", r"\bcyber\b", r"\bdigital\b"),
]
ASSET_CLASS_MAP = {
    "Equity": "Equity", "Equity ACTIVE": "Equity", "Fixed Income": "Fixed Income",
    "Fixed Income ACTIVE": "Fixed Income", "Multi Asset": "Multi Asset",
    "Multi Asset ACTIVE": "Multi Asset", "Commodity": "Commodity",
    "Digital Assets": "Alternative", "Cash ACTIVE": "Other", "Real Estate": "Other",
}

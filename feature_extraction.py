"""Feature extraction helpers for the phishing detection system.

The Flask app imports this module, so it must not train models or write files
at import time. Demo code is kept behind the __main__ guard at the bottom.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

URGENCY_WORDS = [
    "urgent",
    "immediately",
    "suspended",
    "verify",
    "confirm",
    "alert",
    "warning",
    "action required",
    "click here",
    "limited time",
    "expire",
    "access denied",
    "update your",
    "validate",
    "unusual activity",
    "final warning",
    "within 24 hours",
    "failure to act",
    "act now",
]

SUBJECT_KEYWORDS = [
    "account alert",
    "confirm identity",
    "prize winner",
    "security notice",
    "password reset",
    "unusual sign",
    "verify email",
    "locked account",
    "invoice payment",
    "payment failed",
    "security alert",
    "final notice",
]

SENSITIVE_REQUEST_WORDS = [
    "password",
    "pin",
    "otp",
    "one-time code",
    "credit card",
    "debit card",
    "bank details",
    "social security",
    "ssn",
    "login details",
    "security question",
    "seed phrase",
    "private key",
    "card number",
    "account number",
    "routing number",
    "verification code",
    "confirm your identity",
    "verify your identity",
]

HIGH_RISK_EMAIL_PHRASES = [
    "account has been suspended",
    "account will be suspended",
    "verify your account",
    "confirm your account",
    "update your payment",
    "payment failed",
    "unusual activity",
    "login immediately",
    "click here",
    "failure to act",
    "provide your password",
    "enter your password",
    "confirm your identity",
]

LEGITIMATE_CONTEXT_WORDS = [
    "meeting",
    "agenda",
    "minutes",
    "schedule",
    "project",
    "report",
    "attached report",
    "review",
    "team",
    "invoice attached",
    "thank you",
    "regards",
]

KNOWN_BRAND_DOMAINS = {
    "paypal": {"paypal.com"},
    "amazon": {"amazon.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "outlook.com"},
    "google": {"google.com", "gmail.com"},
    "apple": {"apple.com", "icloud.com"},
    "ebay": {"ebay.com"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "netflix": {"netflix.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
}

SUSPICIOUS_URL_WORDS = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "secure",
    "security",
    "account",
    "update",
    "bank",
    "billing",
    "confirm",
    "auth",
    "authentication",
    "password",
    "reset",
    "recovery",
    "wallet",
    "free",
    "bonus",
    "prize",
    "limited",
    "suspend",
    "suspended",
    "unlock",
    "webscr",
    "2fa",
    "support",
]

RISKY_TLDS = {"zip", "mov", "click", "top", "xyz", "tk", "ml", "ga", "cf", "gq"}
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "ow.ly", "t.co", "short.link", "rebrand.ly"]
FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "aol.com"}
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>'\"]+|[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s<>'\"]*)?", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)

LOOKALIKE_REPLACEMENTS = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "$": "s",
})


def extract_url_features(url: str) -> dict[str, int]:
    """Extract structural URL features from a raw URL string."""
    url = url or ""
    extracted = TLD_EXTRACTOR(url)
    ip_pattern = re.compile(r"(http[s]?://)?(\d{1,3}\.){3}\d{1,3}")
    return {
        "url_length": len(url),
        "https": 1 if url.lower().startswith("https") else 0,
        "num_dots": url.count("."),
        "has_ip": 1 if ip_pattern.search(url) else 0,
        "has_hyphen": 1 if "-" in extracted.domain else 0,
        "has_special_chars": 1 if re.search(r"[@%]|//", url) else 0,
        "subdomain_depth": extracted.subdomain.count(".") + 1
        if extracted.subdomain
        else 0,
        "is_shortened": 1 if any(shortener in url.lower() for shortener in SHORTENERS) else 0,
    }


def analyze_url_risk(url: str) -> dict[str, object]:
    """Score a URL using explicit phishing indicators."""
    url = url or ""
    lower_url = url.lower()
    parsed = urlparse(url if "://" in url else f"http://{url}")
    hostname = (parsed.hostname or "").lower()
    extracted = TLD_EXTRACTOR(url)
    registered_domain = ".".join(part for part in [extracted.domain, extracted.suffix] if part).lower()
    domain_text = extracted.domain.lower()
    normalized_domain = domain_text.translate(LOOKALIKE_REPLACEMENTS)
    basic = extract_url_features(url)

    score = 0
    reasons: list[str] = []

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    if basic["has_ip"]:
        add(35, "uses an IP address instead of a normal domain")

    if basic["is_shortened"]:
        add(30, "uses a URL shortener that hides the final destination")

    if parsed.scheme != "https":
        add(12, "does not use HTTPS")

    if len(url) >= 75:
        add(18, "has an unusually long URL")
    elif len(url) >= 54:
        add(8, "has a moderately long URL")

    if "@" in url:
        add(25, "contains @, which can hide the real destination")

    if lower_url.find("//", 8) >= 0:
        add(20, "contains an extra redirect-style //")

    if basic["has_hyphen"]:
        add(12, "uses a hyphenated domain")

    if basic["subdomain_depth"] >= 2:
        add(12, "uses multiple subdomains")

    if extracted.suffix.lower() in RISKY_TLDS:
        add(20, "uses a risky or commonly abused top-level domain")

    if any(char.isdigit() for char in extracted.domain):
        add(16, "uses numbers inside the domain name")

    suspicious_hits = [word for word in SUSPICIOUS_URL_WORDS if word in lower_url]
    if suspicious_hits:
        add(min(38, len(suspicious_hits) * 10), f"contains suspicious words: {', '.join(suspicious_hits[:4])}")

    for brand, allowed_domains in KNOWN_BRAND_DOMAINS.items():
        if brand in lower_url and registered_domain not in allowed_domains:
            add(35, f"mentions {brand} outside its official domain")
            break
        if (
            registered_domain not in allowed_domains
            and brand not in domain_text
            and (brand in normalized_domain or normalized_domain in brand)
            and len(domain_text) >= 4
        ):
            add(35, f"uses a lookalike domain for {brand}")
            break

    if any(token in domain_text for token in ["secure", "login", "verify", "account", "support"]):
        add(14, "places credential-related words directly in the domain")

    if "-" in domain_text and suspicious_hits:
        add(10, "combines a hyphenated domain with credential wording")

    if parsed.query:
        add(10, "contains query parameters often used in tracking or credential links")

    path_depth = len([part for part in parsed.path.split("/") if part])
    if path_depth >= 3:
        add(10, "uses a deep path structure")
    elif path_depth >= 1 and suspicious_hits:
        add(6, "uses a credential-themed path")

    score = min(score, 100)
    return {
        "score": score,
        "reasons": reasons[:6],
        "registered_domain": registered_domain or hostname,
    }


def extract_model_url_features(url: str, feature_names: list[str]) -> dict[str, int]:
    """Map a live URL into the feature names used by phishing.csv.

    The dataset stores many URL checks as -1, 0, or 1. We approximate the
    unavailable host/page checks from the URL text so live predictions are
    still driven by the actual user input instead of a vector of zeros.
    """
    url = url or ""
    lower_url = url.lower()
    parsed = urlparse(url if "://" in url else f"http://{url}")
    hostname = parsed.hostname or ""
    extracted = TLD_EXTRACTOR(url)
    basic = extract_url_features(url)

    known_brands = ["paypal", "amazon", "microsoft", "google", "apple", "bank", "ebay"]

    has_ip = basic["has_ip"] == 1
    is_https = parsed.scheme == "https" or lower_url.startswith("https")
    is_short = basic["is_shortened"] == 1
    has_brand = any(brand in lower_url for brand in known_brands)
    suspicious_word_count = sum(1 for word in SUSPICIOUS_URL_WORDS if word in lower_url)
    has_risky_tld = extracted.suffix.lower() in RISKY_TLDS
    domain_has_digit = any(char.isdigit() for char in extracted.domain)
    many_subdomains = basic["subdomain_depth"] >= 2
    path_depth = len([part for part in parsed.path.split("/") if part])
    has_query = bool(parsed.query)

    mapped = {
        "Index": 0,
        "UsingIP": -1 if has_ip else 1,
        "LongURL": -1 if len(url) >= 75 else 0 if len(url) >= 54 else 1,
        "ShortURL": -1 if is_short else 1,
        "Symbol@": -1 if "@" in url else 1,
        "Redirecting//": -1 if lower_url.find("//", 8) >= 0 else 1,
        "PrefixSuffix-": -1 if basic["has_hyphen"] else 1,
        "SubDomains": -1 if many_subdomains else 0 if basic["subdomain_depth"] == 1 else 1,
        "HTTPS": 1 if is_https else -1,
        "DomainRegLen": -1 if has_risky_tld or domain_has_digit else 1,
        "Favicon": 1,
        "NonStdPort": -1 if parsed.port not in (None, 80, 443) else 1,
        "HTTPSDomainURL": -1 if "https" in hostname.lower() and not is_https else 1,
        "RequestURL": -1 if suspicious_word_count >= 2 or has_query else 1,
        "AnchorURL": -1 if suspicious_word_count >= 2 else 1,
        "LinksInScriptTags": -1 if suspicious_word_count >= 3 else 1,
        "ServerFormHandler": -1 if any(word in lower_url for word in ["login", "verify", "password"]) else 1,
        "InfoEmail": -1 if "mailto:" in lower_url else 1,
        "AbnormalURL": -1 if has_ip or domain_has_digit or has_risky_tld else 1,
        "WebsiteForwarding": -1 if lower_url.find("//", 8) >= 0 else 1,
        "StatusBarCust": 1,
        "DisableRightClick": 1,
        "UsingPopupWindow": 1,
        "IframeRedirection": 1,
        "AgeofDomain": -1 if has_risky_tld or domain_has_digit else 1,
        "DNSRecording": -1 if has_ip or not extracted.suffix else 1,
        "WebsiteTraffic": 1 if has_brand and is_https else -1 if suspicious_word_count >= 2 else 0,
        "PageRank": 1 if has_brand and is_https else -1 if has_risky_tld else 0,
        "GoogleIndex": 1 if has_brand and is_https else -1,
        "LinksPointingToPage": 1 if has_brand else 0,
        "StatsReport": -1 if has_ip or has_risky_tld or suspicious_word_count >= 3 else 1,
    }

    mapped["RequestURL"] = -1 if path_depth >= 3 or has_query else mapped["RequestURL"]
    return {name: mapped.get(name, 0) for name in feature_names}


def extract_email_features(
    email_body: str,
    subject: str = "",
    sender: str = "",
) -> dict[str, int]:
    """Extract simple NLP-style warning features from an email."""
    email_body = email_body or ""
    subject = subject or ""
    sender = sender or ""

    text_lower = email_body.lower()
    subject_lower = subject.lower()
    sender_lower = sender.lower()

    combined_text = f"{subject}\n{email_body}"
    combined_lower = combined_text.lower()
    urls = extract_urls_from_text(email_body)
    urgency_count = sum(1 for word in URGENCY_WORDS if word in combined_lower)
    sensitive_request_count = sum(1 for word in SENSITIVE_REQUEST_WORDS if word in combined_lower)
    brands = ["paypal", "amazon", "microsoft", "google", "apple", "bank", "ebay", "facebook", "instagram", "netflix", "chase"]
    brand_in_body = any(brand in text_lower for brand in brands)
    brand_in_sender = any(brand in sender_lower for brand in brands)
    non_ascii_count = sum(1 for char in email_body if ord(char) > 127)
    sender_check = validate_email_sender(sender, combined_text)
    suspicious_urls = [item for item in analyze_urls_in_email(urls) if item["score"] >= 30]

    return {
        "urgency_count": urgency_count,
        "has_urgency": 1 if urgency_count > 0 else 0,
        "url_count_in_body": len(urls),
        "has_suspicious_links": 1 if suspicious_urls or len(urls) > 2 else 0,
        "suspicious_subject": 1
        if any(keyword in subject_lower for keyword in SUBJECT_KEYWORDS)
        else 0,
        "sender_mismatch": 1 if sender_check["is_suspicious"] or (brand_in_body and not brand_in_sender) else 0,
        "sensitive_request_count": sensitive_request_count,
        "requests_sensitive_info": 1 if sensitive_request_count > 0 else 0,
        "encoding_anomalies": 1 if non_ascii_count > 5 else 0,
        "body_length": len(email_body),
        "has_html": 1 if re.search(r"<[a-z]+>", email_body.lower()) else 0,
        "invalid_sender": 1 if not sender_check["is_valid"] else 0,
        "free_sender_domain": 1 if sender_check["uses_free_domain"] else 0,
        "suspicious_url_count": len(suspicious_urls),
    }


def extract_urls_from_text(text: str) -> list[str]:
    """Extract visible URLs and bare domains from email text."""
    urls = []
    for match in URL_PATTERN.findall(text or ""):
        cleaned = match.rstrip(".,);]}>")
        if "@" in cleaned and not cleaned.lower().startswith(("http://", "https://", "www.")):
            continue
        if "." in cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls[:10]


def analyze_urls_in_email(urls: list[str]) -> list[dict[str, object]]:
    """Run URL risk checks against links found inside an email."""
    findings = []
    for url in urls:
        analysis = analyze_url_risk(url)
        findings.append({
            "url": url,
            "score": analysis["score"],
            "registered_domain": analysis["registered_domain"],
            "reasons": analysis["reasons"],
        })
    return findings


def validate_email_sender(sender: str, text: str = "") -> dict[str, object]:
    """Validate sender format and look for brand/domain mismatch."""
    sender = (sender or "").strip()
    match = EMAIL_PATTERN.search(sender)
    email_address = match.group(0).lower() if match else ""
    domain = email_address.split("@")[-1] if email_address else ""
    extracted = TLD_EXTRACTOR(domain)
    registered_domain = ".".join(part for part in [extracted.domain, extracted.suffix] if part).lower()
    lower_text = (text or "").lower()
    mentioned_brand = next((brand for brand in KNOWN_BRAND_DOMAINS if brand in lower_text), "")

    reasons = []
    if not email_address:
        reasons.append("sender address is missing or not a valid email format")

    uses_free_domain = registered_domain in FREE_EMAIL_DOMAINS
    if uses_free_domain and mentioned_brand:
        reasons.append("brand message comes from a free email domain")

    for brand, allowed_domains in KNOWN_BRAND_DOMAINS.items():
        if brand in lower_text and registered_domain and registered_domain not in allowed_domains:
            reasons.append(f"message mentions {brand} but sender is not an official {brand} domain")
            break

    if registered_domain and any(token in extracted.domain for token in ["secure", "support", "verify", "login", "account"]):
        reasons.append("sender domain contains credential-related wording")

    return {
        "is_valid": bool(email_address),
        "email": email_address,
        "domain": registered_domain,
        "uses_free_domain": uses_free_domain,
        "is_suspicious": bool(reasons),
        "reasons": reasons,
    }


def _confidence_from_score(score: int, prediction: str) -> tuple[float, str]:
    """Map a verdict into the requested confidence bands."""
    if prediction == "PHISHING":
        if score >= 75:
            return min(99.0, 90 + ((score - 75) / 25 * 9)), "HIGH"
        return min(89.0, 70 + ((score - 50) / 25 * 19)), "MEDIUM"

    if prediction == "SUSPICIOUS":
        if score >= 35:
            return min(89.0, 70 + ((score - 35) / 15 * 19)), "MEDIUM"
        return min(69.0, 50 + ((score - 20) / 15 * 19)), "LOW"

    certainty = max(0, 24 - score)
    if score <= 8:
        return min(99.0, 90 + ((8 - score) / 8 * 9)), "HIGH"
    return min(89.0, 70 + (certainty / 16 * 19)), "MEDIUM"


def score_email_risk(
    email_body: str,
    subject: str = "",
    sender: str = "",
    algorithm: str = "random_forest",
) -> dict[str, object]:
    """Classify email text using the available email warning features."""
    features = extract_email_features(email_body, subject, sender)
    sender_check = validate_email_sender(sender, f"{subject}\n{email_body}")
    urls = extract_urls_from_text(email_body)
    url_findings = analyze_urls_in_email(urls)
    combined_text = f"{subject}\n{email_body}"
    combined_lower = combined_text.lower()
    score = 0
    reasons: list[str] = []

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    if features["has_urgency"]:
        add(min(features["urgency_count"], 4) * 10, "uses urgent or threatening language")
    if features["suspicious_subject"]:
        add(16, "subject line matches common phishing wording")
    high_risk_hits = [phrase for phrase in HIGH_RISK_EMAIL_PHRASES if phrase in combined_lower]
    if high_risk_hits:
        add(min(35, len(high_risk_hits) * 14), f"matches phishing wording: {', '.join(high_risk_hits[:3])}")
    if features["requests_sensitive_info"]:
        add(min(30, features["sensitive_request_count"] * 12), "requests sensitive information")
    if features["sender_mismatch"] and not features["invalid_sender"]:
        add(22, "sender identity does not match the message content")
    if features["invalid_sender"]:
        add(20, "sender address is missing or invalid")
    if features["free_sender_domain"] and any(brand in f"{subject} {email_body}".lower() for brand in KNOWN_BRAND_DOMAINS):
        add(16, "brand message comes from a free email domain")
    if features["has_html"]:
        add(6, "contains HTML formatting that can hide links")
    if features["encoding_anomalies"]:
        add(8, "contains unusual characters that may be used to disguise text")
    if 0 < features["body_length"] < 25:
        add(12, "message body is too short to verify safely")
    if features["url_count_in_body"] > 0:
        add(8, "contains links that should be verified before opening")
    if features["url_count_in_body"] > 2:
        add(12, "contains multiple links")

    suspicious_url_findings = [finding for finding in url_findings if finding["score"] >= 30]
    sender_domain = sender_check["domain"]
    external_links = [
        finding
        for finding in url_findings
        if sender_domain and finding["registered_domain"] and finding["registered_domain"] != sender_domain
    ]
    if external_links:
        add(10, "contains a link that does not match the sender domain")

    if suspicious_url_findings:
        top_url = suspicious_url_findings[0]
        add(min(35, 18 + len(suspicious_url_findings) * 5), f"contains suspicious URL: {top_url['registered_domain'] or top_url['url']}")
        for reason in top_url["reasons"][:2]:
            reasons.append(f"linked URL {reason}")

    for reason in sender_check["reasons"][:3]:
        if reason not in reasons:
            reasons.append(reason)

    legitimate_hits = [word for word in LEGITIMATE_CONTEXT_WORDS if word in combined_lower]
    if (
        legitimate_hits
        and not features["requests_sensitive_info"]
        and not suspicious_url_findings
        and features["urgency_count"] == 0
    ):
        score = max(0, score - min(16, len(legitimate_hits) * 4))

    score = min(score, 100)
    if score >= 50:
        prediction = "PHISHING"
        risk_level = "HIGH" if score >= 70 else "MEDIUM"
    elif score >= 20:
        prediction = "SUSPICIOUS"
        risk_level = "MEDIUM"
        if not reasons:
            reasons.append("message has limited warning signs but should be treated with caution")
    else:
        prediction = "LEGITIMATE"
        risk_level = "LOW"
        if not reasons:
            reasons.append("no major phishing indicators were found")

    confidence, confidence_band = _confidence_from_score(score, prediction)

    algorithm_label = {
        "random_forest": "Random Forest + Email risk rules",
        "logistic_regression": "Logistic Regression + Email risk rules",
    }.get(algorithm, "Random Forest + Email risk rules")

    return {
        "prediction": prediction,
        "confidence": f"{confidence:.1f}%",
        "confidence_band": confidence_band,
        "risk_level": risk_level,
        "features": features,
        "risk_score": score,
        "algorithm": algorithm_label,
        "reasons": reasons[:8],
        "sender_validation": sender_check,
        "detected_urls": url_findings,
        "signals_checked": 14 + len(urls),
        "active_signals": len(reasons),
    }


def train_sample_tfidf_vectorizer(output_dir: Path = MODELS_DIR) -> TfidfVectorizer:
    """Train and save the small demo TF-IDF vectorizer from the guide."""
    sample_emails = [
        "Your account has been suspended verify now immediately urgent click here",
        "Dear user please update your password account alert security notice",
        "Meeting scheduled for Monday please review the attached agenda",
        "Your quarterly report is ready for download from the shared drive",
        "Congratulations you have won a prize claim your reward immediately",
    ]

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
    )
    vectorizer.fit(sample_emails)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    return vectorizer


def _run_demo() -> None:
    test_url = "http://paypal-login.update-account.net/secure/verify?token=abc"
    print("=== URL FEATURE EXTRACTION TEST ===")
    for key, value in extract_url_features(test_url).items():
        print(f"  {key}: {value}")

    sample_email = """
Dear Customer, Your account has been SUSPENDED due to unusual activity.
Please CLICK HERE immediately to verify your identity: http://paypa1.com/verify
Failure to act within 24 hours will result in permanent account closure.
"""
    print("=== EMAIL FEATURE EXTRACTION TEST ===")
    for key, value in extract_email_features(
        sample_email,
        subject="URGENT: Account Alert",
        sender="support@paypa1-secure.net",
    ).items():
        print(f"  {key}: {value}")

    vectorizer = train_sample_tfidf_vectorizer()
    features = vectorizer.get_feature_names_out()
    print(f"TF-IDF features saved: {len(features)}")
    print(f"Top 20 features: {features[:20]}")


if __name__ == "__main__":
    _run_demo()

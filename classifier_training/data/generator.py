"""
generator.py — Synthetic labelled and unlabelled sample generation.

v2 improvements:
  - Signal vocabularies expanded to 40-50 entries per class with natural paraphrases
  - Three generation modes: phrase, template, long-phrase
  - Longer documents: 5-10 sentences instead of 3-5
  - 80-sentence filler pool across document types
  - Explicit boundary cases for the three highest-risk decision boundaries
  - N_LABELLED raised to 150, N_UNLABELLED to 50

Labels follow the six-level hierarchy (policy v1.1):
  0  PUBLIC
  1  FOUO
  2  CONFIDENTIAL
  3  PERSONAL_CONFIDENTIAL
  4  HIGHLY_CONFIDENTIAL
  5  PERSONAL_HIGHLY_CONFIDENTIAL
"""

import random
from dataclasses import dataclass
from typing import Optional

CLASSES = [
    "PUBLIC",
    "FOUO",
    "CONFIDENTIAL",
    "PERSONAL_CONFIDENTIAL",
    "HIGHLY_CONFIDENTIAL",
    "PERSONAL_HIGHLY_CONFIDENTIAL",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

NOISE_RATE   = 0.50
N_LABELLED   = 150
N_UNLABELLED = 50

SIGNALS = {
    "PUBLIC": [
        "for immediate release", "press release", "no restriction",
        "authorized for release", "approved for public distribution",
        "public document", "no classification", "unclassified",
        "cleared for public release", "intended for general audiences",
        "available on our website", "open to all stakeholders",
        "freely available", "available to the public",
        "published on the corporate website", "accessible without restriction",
        "open access document", "media contact", "public announcement",
        "external newsletter", "investor relations announcement",
        "published quarterly report", "community update",
        "annual report", "product launch announcement",
        "official statement", "earnings release",
        "regulatory public filing", "published research report",
        "publicly available guidance", "general information only",
        "this document is publicly available", "no handling restrictions apply",
        "suitable for unrestricted distribution", "may be shared freely",
        "intended for broad distribution", "no confidentiality obligation applies",
        "published for external audiences", "approved for external use",
        "for public distribution",
    ],
    "FOUO": [
        "for official use only", "FOUO",
        "not for external distribution", "internal use only",
        "do not distribute externally", "for internal audiences only",
        "internal eyes only", "not for public release",
        "restricted to internal staff", "do not share outside the organisation",
        "internal circulation only", "official use only do not forward",
        "weekly team sync notes", "all-staff communication",
        "internal memo", "org-wide update", "team briefing",
        "internal announcement", "staff newsletter internal",
        "operations update internal", "internal project status update",
        "department status report", "non-sensitive internal report",
        "routine operational update", "internal process documentation",
        "internal training materials", "onboarding documentation internal",
        "meeting notes not for distribution", "headcount planning preliminary",
        "department budget overview non-strategic", "vendor contact list internal",
        "office operations update", "facilities management internal",
        "IT systems update internal", "internal policy reminder",
        "internal calendar and scheduling", "non-critical project milestone update",
        "internal resource allocation overview", "administrative communication",
        "internal directory update", "team performance overview internal",
        "internal expense report summary", "non-sensitive procurement update",
        "internal bulletin board", "staff briefing internal only",
    ],
    "CONFIDENTIAL": [
        "confidential", "strictly confidential", "restricted distribution",
        "company confidential", "confidential do not copy",
        "not for unauthorised disclosure", "confidential business information",
        "commercially sensitive", "in confidence",
        "for authorised recipients only", "non-disclosure agreement",
        "NDA", "contract terms", "partnership agreement",
        "licensing agreement", "settlement agreement", "legal matter",
        "pending litigation", "regulatory filing confidential",
        "compliance matter", "legal opinion",
        "outside counsel memorandum", "business strategy",
        "strategic initiative", "financial projection",
        "revenue forecast", "budget confidential",
        "undisclosed financial results", "pricing proposal",
        "commercial negotiation", "competitive analysis confidential",
        "market entry strategy", "product roadmap confidential",
        "go-to-market strategy", "acquisition preparation",
        "strategic partnership discussion", "under embargo",
        "board presentation confidential", "investor update confidential",
        "intellectual property disclosure", "trade secret disclosure",
        "proprietary business information", "confidential term sheet",
        "restricted to deal team", "commercially sensitive information",
    ],
    "PERSONAL_CONFIDENTIAL": [
        "personally identifiable information", "personal data",
        "data subject", "individual personal record",
        "identifiable personal information", "personal information under GDPR",
        "data protection act personal data", "name and home address",
        "name and phone number", "full name and email address",
        "staff directory personal contact details",
        "employee name and contact details",
        "employee ID and contact details", "personal email address on file",
        "direct personal phone number", "individual contact record",
        "date of birth on file", "national identification number",
        "social security number", "passport number",
        "government issued ID number", "tax identification number",
        "individual demographic data", "individual health information",
        "personal medical record", "health data personal",
        "individual wellness data", "customer personal data",
        "client personal information", "HR personal data record",
        "staff personal details", "employee personal file",
        "personal correspondence", "individual financial record personal",
        "personal bank account details", "individual credit record",
        "private individual information", "personal data processed by HR",
        "consent form personal data", "individual data retention record",
        "personal information restricted to HR", "named individual contact data",
        "individual privacy record", "personal profile data",
    ],
    "HIGHLY_CONFIDENTIAL": [
        "highly confidential", "restricted access only",
        "strictly restricted", "top restricted", "eyes only",
        "need to know basis only", "restricted senior leadership only",
        "board restricted", "maximum restriction",
        "proposed acquisition", "merger valuation", "due diligence",
        "undisclosed M&A transaction", "target company confidential",
        "deal structure confidential", "acquisition financing terms",
        "fairness opinion restricted", "hostile takeover strategy",
        "merger integration plan restricted", "board approval required",
        "board material restricted", "board resolution confidential",
        "governance matter restricted", "shareholder agreement restricted",
        "source code repository", "security credentials",
        "cryptographic key", "authentication tokens",
        "private encryption key", "system access credentials",
        "exploit vulnerability disclosure", "zero-day vulnerability",
        "penetration test findings", "security audit findings",
        "incident response restricted", "legal exposure restricted",
        "litigation strategy restricted", "regulatory enforcement risk",
        "material non-public information", "MNPI restricted",
        "whistleblower report restricted", "IP portfolio valuation restricted",
        "trade secret restricted", "restricted deal information",
    ],
    "PERSONAL_HIGHLY_CONFIDENTIAL": [
        "private highly confidential", "personal data restricted",
        "named individual restricted", "identified person highly confidential",
        "named individual performance review", "employee PIP discussion",
        "performance improvement plan named", "named employee disciplinary record",
        "individual redundancy details", "named executive termination record",
        "individual compensation adjustment", "executive compensation details",
        "executive pay restricted named", "bonus allocation named individual",
        "equity grant named employee", "employee HR investigation personal",
        "named individual grievance file", "personal health record restricted",
        "individual medical assessment confidential",
        "named employee occupational health", "private medical record confidential",
        "named individual disability record", "personal legal proceedings",
        "named individual in litigation", "named individual criminal record",
        "individual legal exposure named", "named beneficiary financial data",
        "individual M&A participant data", "named shareholder restricted",
        "personal tax record restricted", "individual security clearance record",
        "named individual background investigation", "biometric data",
        "personal data security incident named", "named individual audit finding",
        "personal data in board material", "named individual strictly confidential",
        "private individual highly restricted", "personal file board level",
    ],
}

FILLER_GENERAL = [
    "Please review the attached materials at your earliest convenience.",
    "This document has been prepared for the upcoming meeting.",
    "Further details will be shared as the project progresses.",
    "All figures are preliminary and subject to revision.",
    "Kindly ensure appropriate handling in line with company policy.",
    "The team will reconvene next week to discuss next steps.",
    "Distribution is limited to those with a need to know.",
    "Questions should be directed to the document owner.",
    "This supersedes all previous versions of the document.",
    "Please do not forward without prior authorisation.",
    "Recipients are reminded of their confidentiality obligations.",
    "This document should be stored securely when not in use.",
    "Unauthorised disclosure may result in disciplinary action.",
    "Please acknowledge receipt by return.",
    "This communication is intended solely for the named recipient.",
]
FILLER_EMAIL = [
    "Please find attached the document for your review.",
    "I wanted to follow up on our conversation from last week.",
    "Could you please confirm receipt of this message?",
    "Let me know if you have any questions or concerns.",
    "I have copied the relevant team members on this email.",
    "Please treat this communication with appropriate discretion.",
    "I look forward to your response.",
    "This email and any attachments are intended for the addressee only.",
    "If you have received this in error please notify the sender immediately.",
    "Please do not hesitate to reach out if you require further clarification.",
    "I will follow up with a formal written summary after the call.",
    "We will schedule a follow-up meeting to discuss the next steps.",
]
FILLER_REPORT = [
    "This report was prepared by the Strategy and Planning team.",
    "The findings contained herein are based on data available as of the report date.",
    "This document is subject to change without prior notice.",
    "References to third parties are for illustrative purposes only.",
    "The conclusions in this report are provisional pending board sign-off.",
    "All financial figures are in EUR unless otherwise stated.",
    "This report is intended for internal use and should not be circulated externally.",
    "Comparative data from prior periods has been restated for consistency.",
    "This is a working draft and has not been subject to legal review.",
    "The analysis does not constitute legal financial or investment advice.",
    "Version control: see document history for revision log.",
]
FILLER_HR = [
    "This record is maintained in accordance with applicable data protection legislation.",
    "Access to this file is restricted to HR personnel and line management.",
    "The employee has been informed of the contents of this document.",
    "This document forms part of the individual's personnel file.",
    "Retention period: seven years from date of creation.",
    "Any queries regarding this record should be directed to HR Business Partners.",
    "This document has been reviewed and signed by the relevant manager.",
    "The employee was given the opportunity to respond in writing.",
]
FILLER_LEGAL = [
    "This document is protected by legal professional privilege.",
    "Prepared in anticipation of litigation.",
    "Not for disclosure to any third party without prior written consent.",
    "This memorandum reflects the legal position as of the date indicated.",
    "Subject to without-prejudice privilege.",
    "For settlement purposes only not admissible as evidence.",
    "Outside counsel has reviewed and approved this communication.",
]
ALL_FILLER = FILLER_GENERAL + FILLER_EMAIL + FILLER_REPORT + FILLER_HR + FILLER_LEGAL

TEMPLATES = {
    "PUBLIC": [
        "PRESS RELEASE\n\nFor immediate release. {s0}\n\n{f0}\n\n{s1}\n\nMedia contact: press@organisation.com. {f1}",
        "PUBLIC NOTICE\n\n{s0}\n\n{f0} {s1}\n\nThis document is approved for public distribution. {f1}",
        "INVESTOR RELATIONS EARNINGS RELEASE\n\n{s0} {s1}\n\n{f0}\n\nNo restriction applies to this communication. {f1}",
        "EXTERNAL NEWSLETTER\n\n{s0}\n\n{f0} {s1}\n\nAvailable on our website. General information only.",
        "OFFICIAL STATEMENT\n\n{s0}\n\n{f0}\n\n{s1} {f1}\n\nCleared for public release. No handling restrictions apply.",
    ],
    "FOUO": [
        "FOR OFFICIAL USE ONLY\n\nTo: All Staff\nSubject: {s0}\n\n{f0} {s1}\n\nDo not distribute externally. {f1}",
        "INTERNAL MEMO\n\nFrom: Operations\nTo: Team Leads\n\n{s0}\n\n{f0} {s1} {f1}\n\nFor internal use only.",
        "FOUO NOT FOR EXTERNAL DISTRIBUTION\n\n{s0} {s1}\n\n{f0}\n\n{f1} This document is for internal discussion only.",
        "Internal briefing note\n\n{s0}\n\n{f0} {f1}\n\n{s1}\n\nInternal use only. Please do not forward.",
        "WEEKLY TEAM SYNC MEETING NOTES\n\nAttendees: see distribution list\n\n{s0} {s1}\n\n{f0} {f1}\n\nNot for external distribution.",
        "INTERNAL ANNOUNCEMENT\n\nTo: All Staff\n\n{s0}\n\n{f0} {s1}\n\n{f1}\n\nFor official use only. Do not share externally.",
    ],
    "CONFIDENTIAL": [
        "CONFIDENTIAL\n\n{s0}\n\n{f0} {s1}\n\n{f1} This document is confidential and intended solely for authorised recipients.",
        "STRICTLY CONFIDENTIAL\n\nSubject: {s0}\n\n{f0}\n\n{s1} {f1}\n\nUnauthorised disclosure is prohibited.",
        "Company Confidential\n\nTO: Named Recipients\nRE: {s0}\n\n{f0} {s1}\n\n{f1}\n\nRestricted distribution. Do not copy.",
        "CONFIDENTIAL RESTRICTED DISTRIBUTION\n\n{s0} {s1}\n\n{f0}\n\n{f1} Recipients are reminded of their NDA obligations.",
        "BOARD PRESENTATION CONFIDENTIAL\n\nAgenda item: {s0}\n\n{f0} {s1}\n\nFor authorised recipients only. {f1}",
        "COMMERCIALLY SENSITIVE\n\n{s0}\n\n{f0}\n\n{s1} {f1}\n\nThis document is subject to non-disclosure agreement.",
    ],
    "PERSONAL_CONFIDENTIAL": [
        "PERSONAL AND CONFIDENTIAL\n\nRe: {s0}\n\nDear Name,\n\n{f0} {s1}\n\n{f1}\n\nThis communication contains personal data and should be handled accordingly.",
        "FOR OFFICIAL USE ONLY CONTAINS PERSONAL DATA\n\n{s0}\n\n{f0} {s1}\n\n{f1} Access restricted to authorised personnel.",
        "HR RECORD PERSONAL DATA\n\nEmployee: Name redacted\n\n{s0} {s1}\n\n{f0}\n\n{f1}\n\nPersonally identifiable information. Handle in accordance with data protection policy.",
        "STAFF DIRECTORY INTERNAL\n\n{s0}\n\n{f0} {s1}\n\n{f1}\n\nContains personal contact details. Not for external distribution.",
        "PERSONAL DATA RECORD\n\n{s0} {s1}\n\n{f0}\n\n{f1}\n\nThis file contains personal information about an identifiable individual.",
    ],
    "HIGHLY_CONFIDENTIAL": [
        "HIGHLY CONFIDENTIAL RESTRICTED\n\n{s0}\n\n{f0} {s1}\n\n{f1}\n\nAccess restricted to named recipients only. Unauthorised disclosure may constitute a criminal offence.",
        "RESTRICTED BOARD ONLY\n\nProject: REDACTED\n\n{s0} {s1}\n\n{f0}\n\n{f1}\n\nHighly confidential. Eyes only.",
        "HIGHLY CONFIDENTIAL\n\nSubject: {s0}\n\n{f0}\n\n{s1} {f1}\n\nThis document contains material non-public information. Do not distribute.",
        "STRICTLY RESTRICTED\n\nCode name: REDACTED\n\n{s0}\n\n{f0} {s1}\n\nNeed to know basis only. {f1}",
        "MAXIMUM RESTRICTION\n\n{s0} {s1}\n\n{f0}\n\n{f1}\n\nHighly confidential. Board and senior leadership only.",
    ],
    "PERSONAL_HIGHLY_CONFIDENTIAL": [
        "PRIVATE AND HIGHLY CONFIDENTIAL\n\nRe: {s0}\n\n{f0} {s1}\n\n{f1}\n\nThis document contains sensitive personal data combined with highly confidential information. Access strictly controlled.",
        "STRICTLY CONFIDENTIAL NAMED INDIVIDUAL\n\nEmployee: Name on file\n\n{s0} {s1}\n\n{f0}\n\n{f1}\n\nPrivate and highly confidential. HR use only.",
        "PERSONAL HIGHLY CONFIDENTIAL\n\nSubject: {s0}\n\n{f0} {s1}\n\n{f1}\n\nContains personal data in a restricted context. Authorised recipients only.",
        "PRIVATE HIGHLY CONFIDENTIAL RESTRICTED\n\n{s0}\n\n{f0}\n\n{s1} {f1}\n\nNamed individual data. Maximum restriction applies.",
        "EYES ONLY PERSONAL DATA\n\n{s0} {s1}\n\n{f0}\n\n{f1}\n\nContains named individual information at the highest sensitivity level.",
    ],
}

BOUNDARY_CASES: list[tuple[str, str]] = [
    # FOUO + PII -> floor at PERSONAL_CONFIDENTIAL
    (
        "FOR OFFICIAL USE ONLY\n\nStaff directory update EMEA region.\n"
        "John Martinez | j.martinez@org.com | +33 6 12 34 56 78 | Paris office.\n"
        "Aisha Nkomo | a.nkomo@org.com | +44 7700 900123 | London office.\n"
        "For internal use only. Contains personal contact details.",
        "PERSONAL_CONFIDENTIAL",
    ),
    (
        "FOUO Internal HR Update.\n"
        "Employee census data attached. Includes full names, dates of birth, and home postcodes for all EMEA staff. "
        "Do not distribute externally. For official use only.",
        "PERSONAL_CONFIDENTIAL",
    ),
    (
        "Internal Memo For Official Use Only.\n"
        "Attached is the updated contact list for the London office. "
        "It includes personal email addresses and direct phone numbers for all team members. "
        "Please treat this information with appropriate care.",
        "PERSONAL_CONFIDENTIAL",
    ),
    (
        "FOUO Onboarding Documentation.\n"
        "New hire details: Name: Sarah Chen, DOB: 14 March 1990, Address: 12 Rue de Rivoli Paris. "
        "Employee ID: EMP-00421. Start date: 1 April 2024. Not for external distribution.",
        "PERSONAL_CONFIDENTIAL",
    ),
    (
        "FOR OFFICIAL USE ONLY. Customer contact record update. "
        "Includes customer full names, personal email addresses, and billing addresses. "
        "Internal use only. Contains personally identifiable information.",
        "PERSONAL_CONFIDENTIAL",
    ),
    (
        "FOUO Internal staff update.\n"
        "The following employees have been allocated to the new project team: "
        "Maria Garcia (m.garcia@org.com, DOB 5 Feb 1988), Tom Nguyen (t.nguyen@org.com, DOB 12 Sep 1991). "
        "Not for external distribution. Personal data included.",
        "PERSONAL_CONFIDENTIAL",
    ),
    # FOUO vs CONFIDENTIAL
    (
        "NOT FOR EXTERNAL DISTRIBUTION.\n"
        "Q3 pipeline overview. Total pipeline value: 2.1M EUR. Forecast close rate: 62 percent. "
        "Key accounts reviewed with regional leads. This document is for internal use only.",
        "FOUO",
    ),
    (
        "Internal Use Only.\n"
        "Department budget overview for FY2024. Total operating budget: 4.2M EUR. "
        "Headcount: 47 FTEs. No strategic implications. For internal planning purposes only.",
        "FOUO",
    ),
    (
        "CONFIDENTIAL.\n"
        "Five-year financial projection strategic planning cycle. "
        "EBITDA forecast: 120M EUR by FY2028. Revenue CAGR: 14 percent. "
        "This document contains forward-looking statements and is subject to NDA. "
        "Unauthorised disclosure could cause significant financial harm.",
        "CONFIDENTIAL",
    ),
    (
        "STRICTLY CONFIDENTIAL.\n"
        "Proposed strategic partnership with redacted Partner Name. "
        "Term sheet attached. Exclusivity period: 90 days. "
        "Do not discuss outside the deal team. NDA in place.",
        "CONFIDENTIAL",
    ),
    (
        "For Official Use Only. Internal project status update Q3. "
        "Milestones on track. No blockers reported. Next review: 15 October. "
        "Routine update for internal planning purposes.",
        "FOUO",
    ),
    (
        "INTERNAL MEMO. For Official Use Only.\n"
        "Headcount report for Q3. Current headcount: 412. "
        "Proposed additions: 18 roles across Engineering and Sales. "
        "These figures are for internal planning only and are not final.",
        "FOUO",
    ),
    # CONFIDENTIAL vs HIGHLY_CONFIDENTIAL
    (
        "HIGHLY CONFIDENTIAL.\n"
        "Proposed acquisition of REDACTED at a valuation of 340M EUR. "
        "Transaction remains subject to board approval. "
        "Advisors instructed to proceed with due diligence on the target IP portfolio. "
        "Material non-public information. Do not disclose.",
        "HIGHLY_CONFIDENTIAL",
    ),
    (
        "CONFIDENTIAL. Annual revenue forecast internal use. "
        "Projected revenue for FY2025: 85M EUR. "
        "This is a planning document and does not constitute financial guidance. "
        "For authorised internal recipients only.",
        "CONFIDENTIAL",
    ),
    (
        "RESTRICTED BOARD ONLY.\n"
        "Security audit findings: critical vulnerabilities identified in production systems. "
        "Authentication credentials exposed. Immediate remediation required. "
        "Highly confidential. Do not share outside the security team and board.",
        "HIGHLY_CONFIDENTIAL",
    ),
    (
        "STRICTLY CONFIDENTIAL.\n"
        "Pending litigation matter employment tribunal. "
        "Legal counsel preliminary assessment attached. "
        "Without-prejudice communication. For authorised recipients only.",
        "CONFIDENTIAL",
    ),
    (
        "HIGHLY CONFIDENTIAL. Zero-day vulnerability disclosed by external researcher. "
        "Affects production authentication layer. "
        "Patch timeline: 72 hours. Do not discuss outside incident response team. "
        "Material non-public information.",
        "HIGHLY_CONFIDENTIAL",
    ),
    # PERSONAL_CONFIDENTIAL vs PERSONAL_HIGHLY_CONFIDENTIAL
    (
        "PERSONAL AND CONFIDENTIAL.\n"
        "Employee record: James Osei, DOB 22 Jun 1985. "
        "Contact: j.osei@org.com, +44 7700 900456. Role: Senior Analyst. "
        "Personal data. Handle in line with data protection policy.",
        "PERSONAL_CONFIDENTIAL",
    ),
    (
        "STRICTLY CONFIDENTIAL NAMED INDIVIDUAL.\n"
        "Performance review: James Osei, EMP-00512. "
        "Rating: Underperforming. PIP initiated 1 March 2024. "
        "Compensation review on hold pending outcome. "
        "Named individual HR record. Highly confidential.",
        "PERSONAL_HIGHLY_CONFIDENTIAL",
    ),
    (
        "PRIVATE AND HIGHLY CONFIDENTIAL.\n"
        "Executive compensation review FY2024. "
        "Name: Claire Fontaine. Total package: 420000 EUR. "
        "LTI grant: 15000 shares. Retention bonus: 50000 EUR. "
        "For remuneration committee only. Strictly restricted.",
        "PERSONAL_HIGHLY_CONFIDENTIAL",
    ),
    (
        "PERSONAL CONFIDENTIAL.\n"
        "Customer data subject access request. "
        "Name: Mehmet Yilmaz. Account: CUS-00891. "
        "Personal data disclosed in accordance with GDPR Article 15. "
        "Data subject has been notified.",
        "PERSONAL_CONFIDENTIAL",
    ),
    (
        "PRIVATE HIGHLY CONFIDENTIAL.\n"
        "Named individual: David Okafor. Employee PIP discussion initiated. "
        "Compensation adjustment: minus 10 percent pending review. "
        "Disciplinary record attached. For HR and line manager only. "
        "Named individual highly confidential.",
        "PERSONAL_HIGHLY_CONFIDENTIAL",
    ),
    (
        "PERSONAL AND CONFIDENTIAL.\n"
        "GDPR data subject access request response. "
        "Requestor: Ana Costa. Personal data categories disclosed: contact details, employment history. "
        "No highly confidential data included in scope of this request.",
        "PERSONAL_CONFIDENTIAL",
    ),
]


@dataclass
class Sample:
    text: str
    label: Optional[str]
    label_idx: Optional[int]


def _phrase_sample(cls: str, rng: random.Random) -> str:
    n_signals = rng.randint(3, 6)
    signals = rng.sample(SIGNALS[cls], min(n_signals, len(SIGNALS[cls])))
    if rng.random() < NOISE_RATE:
        neighbour = rng.choice([c for c in CLASSES if c != cls])
        signals.append(rng.choice(SIGNALS[neighbour]))
        rng.shuffle(signals)
    fillers = rng.sample(ALL_FILLER, rng.randint(3, 5))
    parts = signals + fillers
    rng.shuffle(parts)
    return " ".join(parts).capitalize() + "."


def _template_sample(cls: str, rng: random.Random) -> str:
    template = rng.choice(TEMPLATES[cls])
    sigs = rng.sample(SIGNALS[cls], min(4, len(SIGNALS[cls])))
    fils = rng.sample(ALL_FILLER, 2)
    text = template.format(
        s0=sigs[0], s1=sigs[1] if len(sigs) > 1 else sigs[0],
        f0=fils[0], f1=fils[1],
    )
    if rng.random() < NOISE_RATE * 0.4:
        neighbour = rng.choice([c for c in CLASSES if c != cls])
        text += " Note: " + rng.choice(SIGNALS[neighbour]) + "."
    return text


def _build_text(cls: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.40:
        return _phrase_sample(cls, rng)
    elif r < 0.80:
        return _template_sample(cls, rng)
    else:
        # Long pure phrase
        signals = rng.sample(SIGNALS[cls], min(rng.randint(4, 7), len(SIGNALS[cls])))
        fillers = rng.sample(ALL_FILLER, min(rng.randint(4, 6), len(ALL_FILLER)))
        parts = signals + fillers
        rng.shuffle(parts)
        return " ".join(parts).capitalize() + "."


def generate(
    n_labelled: int = N_LABELLED,
    n_unlabelled: int = N_UNLABELLED,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    labelled, unlabelled = [], []

    for text, cls in BOUNDARY_CASES:
        labelled.append(Sample(text=text, label=cls, label_idx=CLASS_TO_IDX[cls]))

    for cls in CLASSES:
        idx = CLASS_TO_IDX[cls]
        for _ in range(n_labelled):
            labelled.append(Sample(text=_build_text(cls, rng), label=cls, label_idx=idx))
        for _ in range(n_unlabelled):
            unlabelled.append(Sample(text=_build_text(cls, rng), label=None, label_idx=None))

    rng.shuffle(labelled)
    rng.shuffle(unlabelled)
    return labelled, unlabelled


if __name__ == "__main__":
    labelled, unlabelled = generate()
    print(f"Labelled:   {len(labelled)} samples  (incl. {len(BOUNDARY_CASES)} boundary cases)")
    print(f"Unlabelled: {len(unlabelled)} samples")
    print(f"\nClass distribution:")
    from collections import Counter
    c = Counter(s.label for s in labelled)
    for cls in CLASSES:
        print(f"  {cls:35s}: {c[cls]}")
    print(f"  {'BOUNDARY (across classes)':35s}: {len(BOUNDARY_CASES)}")

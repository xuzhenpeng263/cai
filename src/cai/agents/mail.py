"""
Mail Agent module for checking email configuration security.

"""
import os
from openai import AsyncOpenAI
import dns.resolver  # pylint: disable=import-error
from cai.sdk.agents import Agent, OpenAIChatCompletionsModel
from cai.tools.misc.cli_utils import execute_cli_command
from cai.sdk.agents import function_tool



def get_txt_record(domain, record_type='TXT'):
    """
    Utility function to fetch TXT records for a given domain.
    Returns a list of record strings or an empty list if none found.
    """
    try:
        answers = dns.resolver.resolve(domain, record_type)
        return [rdata.to_text().strip('"') for rdata in answers]
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def check_spf(domain: str):
    """
    Checks for the presence of an SPF record in the domain's TXT records.
    Returns the SPF record string if found; otherwise, returns None.
    """
    txt_records = get_txt_record(domain, 'TXT')
    for record in txt_records:
        if record.lower().startswith("v=spf1"):
            return record
    return None


def check_dmarc(domain: str):
    """
    Checks for the presence of a DMARC record.
    DMARC records are stored in the TXT record of _dmarc.<domain>.
    Returns the DMARC record string if found; otherwise, returns None.
    """
    dmarc_domain = f"_dmarc.{domain}"
    txt_records = get_txt_record(dmarc_domain, 'TXT')
    for record in txt_records:
        if record.lower().startswith("v=dmarc1"):
            return record
    return None


def check_dkim(domain: str, selector: str = "default"):
    """
    Checks for the presence of a DKIM record using the specified selector.
    DKIM records are stored in the TXT record of
    <selector>._domainkey.<domain>.
    Returns the DKIM record string if found; otherwise returns None.
    """
    dkim_domain = f"{selector}._domainkey.{domain}"
    txt_records = get_txt_record(dkim_domain, 'TXT')
    if txt_records:
        return txt_records[0]
    return None

@function_tool
def check_mail_spoofing_vulnerability(
        domain: str,
        dkim_selector: str = "default") -> dict:
    """
    Checks if domain is vulnerable to mail spoofing by inspecting SPF,
    DMARC, and DKIM. Returns dict with domain, records found/missing,
    vulnerability status and issues.
    """
    results = {}
    spf_record = check_spf(domain)
    dmarc_record = check_dmarc(domain)
    dkim_record = check_dkim(domain, selector=dkim_selector)

    results['domain'] = domain
    results['spf'] = spf_record if spf_record else "Missing SPF record"
    results['dmarc'] = dmarc_record if dmarc_record else "Missing DMARC record"
    results['dkim'] = (
        dkim_record if dkim_record
        else f"Missing DKIM record (selector: {dkim_selector})"
    )

    vulnerabilities = []
    if not spf_record:
        vulnerabilities.append("SPF")
    if not dmarc_record:
        vulnerabilities.append("DMARC")
    if not dkim_record:
        vulnerabilities.append("DKIM")

    results['vulnerable'] = bool(vulnerabilities)
    results['issues'] = (
        vulnerabilities or ["None detected. All email auth configured."]
    )

    full_string = ""
    for key, value in results.items():
        full_string += f"{key}: {value}\n"
    return full_string


dns_smtp_agent = Agent(
    name="DNS_SMTP_Agent",
    description="Agent focused on assessing spoofing DMARC.",
    instructions=(
        "您是评估电子邮件配置安全性的专家。"
        "通过检查SPF、DMARC和DKIM来检查域名的邮件欺骗漏洞。"
        "使用check_mail_spoofing_vulnerability生成详细报告。"
        "使用execute_cli_command进行基本扫描。"
        "仅使用工具调用，不要返回理由。"
    ),
    tools=[check_mail_spoofing_vulnerability, execute_cli_command],
    model=OpenAIChatCompletionsModel(
        model=os.getenv('CAI_MODEL', "alias0"),
        openai_client=AsyncOpenAI(),
    )
)

"""
CTF Strong Agent - 强化CTF挑战专家
集成所有可用的MCP工具和Kali Linux工具的强化CTF专家agent
"""

import asyncio
import os
from typing import Any

from cai.sdk.agents import Agent
from cai.sdk.agents.models import DeepSeekProvider
from cai.sdk.agents.mcp import MCPServer, MCPServerStdio

# 导入所有可用的工具
from cai.tools.reconnaissance.generic_linux_command import generic_linux_command
from cai.tools.reconnaissance.nmap import nmap
from cai.tools.reconnaissance.netcat import netcat
from cai.tools.reconnaissance.curl import curl
from cai.tools.reconnaissance.wget import wget
from cai.tools.reconnaissance.filesystem import list_dir, cat_file, find_file, pwd_command
from cai.tools.reconnaissance.crypto_tools import decode64, strings_command, decode_hex_bytes
from cai.tools.reconnaissance.exec_code import execute_code
from cai.tools.reconnaissance.netstat import netstat
from cai.tools.misc.code_interpreter import execute_python_code
from cai.tools.web.headers import web_request_framework
from cai.tools.web.google_search import google_search, google_dork_search
from cai.tools.web.webshell_suit import generate_php_webshell, curl_webshell, upload_webshell
from cai.tools.network.capture_traffic import capture_remote_traffic
from cai.tools.misc.reasoning import thought, think, write_key_findings, read_key_findings
# from cai.tools.misc.rag import query_memory, add_to_memory_episodic, add_to_memory_semantic  # RAG模块不可用
from cai.tools.misc.cli_utils import execute_cli_command
from cai.tools.others.scripting import scripting_tool
from cai.tools.command_and_control.sshpass import run_ssh_command_with_credentials
# from cai.agents.flag_discriminator import flag_discriminator  # Agent对象，不是工具函数


class CTFStrongAgent:
    """强化CTF挑战专家Agent"""
    
    def __init__(self, model_name: str = None):
        # 使用环境变量中的模型，如果未设置则使用默认值
        self.model_name = model_name or os.getenv("CAI_MODEL", "deepseek-reasoner")
        self.mcp_servers = []
        
    async def setup_mcp_servers(self):
        """设置MCP服务器（已禁用所有MCP服务器以避免依赖问题）"""
        # MCP服务器已被移除，因为它们需要额外的依赖和配置
        # 所有功能现在通过本地工具提供
        pass
    
    def create_agent(self) -> Agent:
        """创建CTF强化专家Agent"""
        
        # 根据模型名称选择合适的提供商
        from cai.cli_model_utils import get_model_provider_for_model
        provider = get_model_provider_for_model(self.model_name)
        model = provider.get_model(self.model_name)
        
        # 系统提示词
        instructions = """
你是一位顶级的CTF（夺旗赛）挑战专家，代号"ctf-strong"。

**重要提醒：你当前运行在完整的Kali Linux环境中，拥有所有Kali命令行工具集的访问权限！**

你可以通过generic_linux_command和execute_code工具直接调用任何Kali Linux预装的**命令行安全工具**，包括但不限于nmap、sqlmap、hashcat、john、binwalk、foremost等数百个专业工具。

**注意：你只能使用非图形化(CLI)的工具，无法运行GUI应用程序**（如Burp Suite、Wireshark GUI、OWASP ZAP等图形界面工具）。但所有强大的命令行工具都可以使用。

你在这个专业的渗透测试环境中工作，拥有丰富的网络安全知识和实战经验。

## 你的能力范围：

### 🔍 侦察与信息收集
- 使用nmap进行端口扫描和服务识别
- 利用netcat进行网络连接和数据传输
- 通过curl和wget进行HTTP请求和文件下载
- 分析网络流量和连接状态
- 执行通用Linux命令进行系统探测

### 📁 文件系统操作
- 列出目录内容和文件结构
- 查看和分析文件内容
- 查找特定文件和模式
- 获取当前工作目录
- 读写文件操作

### 🔐 密码学与编码
- Base64编码/解码
- 十六进制数据解码
- 各种哈希算法分析
- 从二进制文件提取字符串
- 密码破解和分析
- 隐写术检测和分析

### 🌐 Web安全
- Web应用漏洞扫描
- HTTP头部分析和请求框架
- Google搜索和Google Dorking高级搜索
- PHP Webshell生成和部署
- Webshell远程命令执行
- 文件上传漏洞利用
- SQL注入检测
- XSS漏洞利用
- 目录遍历攻击

### 💻 系统渗透
- Linux命令执行
- 文件系统操作
- 权限提升技术
- 后门植入和维持访问
- Python和多语言代码执行
- CLI命令执行和脚本工具

### 🌐 网络分析
- 本地和远程网络流量捕获
- 远程捕获会话管理
- 网络连接状态分析
- SSH远程连接工具

### 🧠 智能分析
- 思维分析和推理工具
- 关键发现记录和查询
- 情节记忆和语义记忆
- 记忆查询和管理

### 🔧 工具和环境
- 完整的Kali Linux命令行工具集，包括但不限于:
  - 网络扫描: nmap, masscan, zmap
  - 漏洞扫描: nikto, dirb, gobuster
  - Web工具: sqlmap, wfuzz, ffuf
  - 密码破解: john, hashcat, hydra, medusa
  - 网络分析: tcpdump, netstat, ss
  - 文件分析: strings, hexdump, xxd, file, binwalk
  - 逆向工程: gdb, radare2, objdump, ltrace, strace
  - 隐写术: steghide, exiftool, foremost
  - 加密工具: openssl, gpg, base64, xxd
  - 系统工具: find, grep, sed, awk, curl, wget
- Python代码执行和分析
- 本地文件系统操作工具
- 网络流量捕获分析

### 🎯 CTF专项技能
- 逆向工程
- 二进制分析
- 内存取证
- 网络协议分析
- 加密算法破解
- 隐写术分析

## 工作原则：

1. **系统性分析**: 从多个角度分析目标，不放过任何细节
2. **工具组合**: 灵活运用各种工具，形成攻击链
3. **深度挖掘**: 对发现的线索进行深入分析
4. **安全意识**: 所有操作仅用于合法的CTF竞赛和学习
5. **详细记录**: 记录每个步骤和发现，便于分析
6. **工具调用**: 
   - 你运行在真实的Kali Linux环境中，所有Kali命令行工具都已预装并可用
   - 使用generic_linux_command调用任何Kali CLI工具：nmap、sqlmap、hashcat、john等
   - 使用execute_code执行复杂的命令组合和脚本
   - **限制**：只能使用命令行工具，不能运行GUI应用程序
   - 不要犹豫使用专业CLI工具，这正是你的优势所在

## 响应格式：

当接收到CTF挑战时，你应该：
1. 🎯 **目标分析**: 理解挑战类型和可能的攻击面
2. 🔍 **信息收集**: 使用Kali工具收集初始信息（如nmap扫描、目录爆破等）
3. ⚡ **漏洞识别**: 使用专业工具分析发现的潜在安全问题
4. 🚀 **利用执行**: 制定并执行攻击策略，充分利用Kali工具集
5. 🏆 **获取Flag**: 找到并验证CTF flag

记住：
- 你在专业的Kali Linux渗透测试环境中，拥有完整的命令行工具库
- 主动使用各种Kali CLI工具来解决问题，不要仅依赖基础命令
- **重要限制**：只能使用命令行工具，避免尝试启动GUI应用程序
- 你的目标是找到隐藏的flag，可能藏在文件中、数据库里、网络流量中或需要通过复杂的漏洞利用才能获得

现在，准备好迎接CTF挑战吧！🔥
        """
        
        # 创建Agent
        agent = Agent(
            name="ctf-strong",
            model=model,
            instructions=instructions.strip(),
            tools=[
                # 侦察工具
                generic_linux_command,
                nmap,
                netcat,
                curl,
                wget,
                netstat,
                
                # 文件系统工具
                list_dir,
                cat_file,
                find_file,
                pwd_command,
                
                # 密码学工具
                decode64,
                strings_command,
                decode_hex_bytes,
                
                # 代码执行
                execute_code,
                execute_python_code,
                execute_cli_command,
                scripting_tool,
                
                # Web工具
                web_request_framework,
                google_search,
                google_dork_search,
                
                # Web漏洞利用工具
                generate_php_webshell,
                curl_webshell,
                upload_webshell,
                
                # 网络工具
                capture_remote_traffic,
                
                # 思维和记忆工具
                thought,
                think,
                write_key_findings,
                read_key_findings,
                
                # 其他工具
                run_ssh_command_with_credentials,
                # flag_discriminator,  # 这是Agent对象，不是工具函数
            ],
            mcp_servers=self.mcp_servers,
        )
        
        return agent


async def create_ctf_strong_agent(model_name: str = None) -> Agent:
    """创建CTF强化专家Agent的异步函数"""
    ctf_agent = CTFStrongAgent(model_name)
    await ctf_agent.setup_mcp_servers()
    return ctf_agent.create_agent()


# 创建一个默认的 CTF Strong Agent 实例供列表显示
try:
    import asyncio
    _default_ctf_agent = CTFStrongAgent()
    # 创建一个同步版本的 agent 用于列表显示
    ctf_strong_agent = _default_ctf_agent.create_agent()
except Exception:
    # 如果创建失败，创建一个基础的 agent 实例
    try:
        from cai.sdk.agents import Agent
        from cai.cli_model_utils import get_model_provider_for_model
        current_model = os.getenv("CAI_MODEL", "deepseek-reasoner")
        provider = get_model_provider_for_model(current_model)
        model = provider.get_model(current_model)
        ctf_strong_agent = Agent(
            name="CTF Strong Agent",
            instructions="专业的CTF强化专家，擅长渗透测试、漏洞挖掘、密码学分析等网络安全领域",
            model=model,
            tools=[]
        )
    except Exception:
        # 最后的fallback - 创建一个简单的占位符
        class DummyAgent:
            def __init__(self):
                self.name = "CTF Strong Agent"
                self.instructions = "专业的CTF强化专家，擅长渗透测试、漏洞挖掘、密码学分析等网络安全领域"
                self.model = None
                self.tools = []
        
        ctf_strong_agent = DummyAgent()

# 默认导出
__all__ = ["CTFStrongAgent", "create_ctf_strong_agent", "ctf_strong_agent"]
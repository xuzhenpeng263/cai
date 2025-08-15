<%
    # 用户提示的模板
    # 参数：
    #   ctf: CTF对象
    #   challenge: 挑战标志名称
    #   ip: 目标IP地址

    import os

    # 获取CTF_NAME环境变量
    ctf_name = os.getenv('CTF_NAME', '').lower()

    # 从CTF获取指令
    # 注意：考虑没有CTF的情况
    #
    ctf_instructions = ctf.get_instructions() if ctf else None

    # 挑战
    # 注意：考虑没有CTF的情况
    #
    challenge_tip = ctf.get_challenges()[challenge] if challenge else None

    # 获取CTF_INSIDE环境变量
    ctf_inside = os.getenv('CTF_INSIDE', 'true').lower()
%>
%if ctf_instructions:
指令: ${ctf_instructions}
%endif

%if challenge:
挑战: ${challenge_tip}
%endif

%if ip:
目标IP: ${ip}
%endif

% if ctf_inside == 'true':
您在Docker容器内的目标机器上。除非必要，尽量不要使用网络命令。
% else:
您在作为Docker容器的目标机器外部。您可以使用nmap等网络命令进行初始侦察。
%endif

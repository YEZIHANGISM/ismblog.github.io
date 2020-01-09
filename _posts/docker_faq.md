---
title: "Docker FAQ"
categories:
  - Docker
tags:
  - docker
---

# 启动容器报错
宿主机是win10，安装docker后，拉取了一个centos镜像，又在镜像上安装docker（禁止套娃），安装docker-ce时报错

    Failed to get D-Bus connection: Operation not permitted
## 解决办法
以下方式生成容器
    
    docker run -d --privileged=true -e "container=docker" container /usr/sbin/init
进入容器
    
    docker exec -it container /bin/bash
如果容器已经生成，只能重新从镜像根据上一条命令生成容器了。
# Docker服务未启动
进入centos中，运行docker命令报错

    # docker run hello-world
    Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
## 解决办法
重启daemon和docker.service

    systemctl daemon-reload
    systemctl restart docker.service
运行第二条命令时报错

    Job for docker.service failed because the control process exited with error code. See "systemctl status docker.service" and "journalctl -xe" for details.
根据提示输入命令
    
    systemctl status docker.service
出现如下信息

    ● docker.service - Docker Application Container Engine
    Loaded: loaded (/usr/lib/systemd/system/docker.service; disabled; vendor preset: disabled)
    Active: failed (Result: start-limit) since Tue 2020-01-07 01:03:24 UTC; 49s ago
        Docs: https://docs.docker.com
    Process: 758 ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock (code=exited, status=1/FAILURE)
    Main PID: 758 (code=exited, status=1/FAILURE)

    Jan 07 01:03:22 cc019bed9d66 systemd[1]: Failed to start Docker Application Container Engine.
    Jan 07 01:03:22 cc019bed9d66 systemd[1]: Unit docker.service entered failed state.
    Jan 07 01:03:22 cc019bed9d66 systemd[1]: docker.service failed.
    Jan 07 01:03:24 cc019bed9d66 systemd[1]: docker.service holdoff time over, scheduling restart.
    Jan 07 01:03:24 cc019bed9d66 systemd[1]: Stopped Docker Application Container Engine.
    Jan 07 01:03:24 cc019bed9d66 systemd[1]: start request repeated too quickly for docker.service
    Jan 07 01:03:24 cc019bed9d66 systemd[1]: Failed to start Docker Application Container Engine.
    Jan 07 01:03:24 cc019bed9d66 systemd[1]: Unit docker.service entered failed state.
    Jan 07 01:03:24 cc019bed9d66 systemd[1]: docker.service failed.
看不出来错误出在哪里，直接运行dockerd

    # dockerd
    INFO[2020-01-07T01:05:31.603772600Z] Starting up
    INFO[2020-01-07T01:05:31.607475700Z] parsed scheme: "unix"                         module=grpc
    INFO[2020-01-07T01:05:31.607632300Z] scheme "unix" not registered, fallback to default scheme  module=grpc
    INFO[2020-01-07T01:05:31.607702300Z] ccResolverWrapper: sending update to cc: {[{unix:///run/containerd/containerd.sock 0  <nil>}] <nil>}  module=grpc
    INFO[2020-01-07T01:05:31.607749300Z] ClientConn switching balancer to "pick_first"  module=grpc
    INFO[2020-01-07T01:05:31.610274600Z] parsed scheme: "unix"                         module=grpc
    INFO[2020-01-07T01:05:31.610376900Z] scheme "unix" not registered, fallback to default scheme  module=grpc
    INFO[2020-01-07T01:05:31.610406400Z] ccResolverWrapper: sending update to cc: {[{unix:///run/containerd/containerd.sock 0  <nil>}] <nil>}  module=grpc
    INFO[2020-01-07T01:05:31.611694200Z] ClientConn switching balancer to "pick_first"  module=grpc
    WARN[2020-01-07T01:05:31.628316400Z] Usage of loopback devices is strongly discouraged for production use. Please use `--storage-opt dm.thinpooldev` or use `man dockerd` to refer to dm.thinpooldev section.  
    storage-driver=devicemapper
    WARN[2020-01-07T01:05:31.847038500Z] XFS is not supported in your system (exec: "mkfs.xfs": executable file not found in $PATH). Defaulting to ext4 filesystem  storage-driver=devicemapper
    INFO[2020-01-07T01:05:31.851265700Z] Creating filesystem ext4 on device docker-0:98-59386-base, mkfs args: [/dev/mapper/docker-0:98-59386-base]  storage-driver=devicemapper
    INFO[2020-01-07T01:05:31.851792400Z] Error while creating filesystem ext4 on device docker-0:98-59386-base: exec: "mkfs.ext4": executable file not found in $PATH  storage-driver=devicemapper
    ERRO[2020-01-07T01:05:31.851883500Z] [graphdriver] prior storage driver devicemapper failed: exec: "mkfs.ext4": executable file not found in $PATH
    failed to start daemon: error initializing graphdriver: exec: "mkfs.ext4": executable file not found in $PATH
最后一条，提示缺少**mkfs.ext4**，使用以下命令安装

    yum install -y e4fsprogs
    # 加载ext4模块
    modprobe ext4
重启docker服务

    systemctl restart docker
# 创建镜像时提示空间不足
        no space left on device
## 解决办法
查看docker磁盘信息
        
        docker system df
        TYPE                TOTAL               ACTIVE              SIZE                RECLAIMABLE
        Images              11                  1                   2.43GB              1.912GB (78%)
        Containers          1                   1                   109.6GB             0B (0%)
        Local Volumes       0                   0                   0B                  0B
        Build Cache         0                   0                   0B                  0B
### 清理无用的系统空间
        
        docker system prune
### 清理无用的卷
        docker volume prune
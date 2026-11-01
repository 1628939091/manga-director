# 评测链路零第三方依赖，所以基础镜像就够了。
# 唯一例外是音频 demo（edge-tts / soundfile），那部分不进这个镜像。
FROM python:3.11-slim

WORKDIR /app

# 先拷代码，再跑测试：测试不过就不构建，等于把 CI 前置到镜像里
COPY mdh/ ./mdh/
COPY web/ ./web/
COPY data/ ./data/
COPY tests/ ./tests/
COPY check_lines.py check_scenes.py run_demo.py run_two_layer.py workbench.py ./

RUN python -m unittest discover -s tests

EXPOSE 8770 8771 8772 8773 8774

# 默认起主工作台；其它服务用 docker run <image> python web/xxx.py 覆盖
CMD ["python", "web/server.py"]

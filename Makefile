.PHONY: help test check serve pair annotate real demo workbench docker run

PY ?= python

help:
	@echo "make test       跑全部单测（54 个）"
	@echo "make check      跑两个数据体检脚本（台词库 / 场景）"
	@echo "make serve      起主工作台  http://127.0.0.1:8770"
	@echo "make pair       起参数对比  http://127.0.0.1:8774/pair"
	@echo "make annotate   起标注台    http://127.0.0.1:8771"
	@echo "make workbench  工作台 CLI：剧本 -> 分镜 + 三种导出"

test:
	$(PY) -m unittest discover -s tests

check:
	$(PY) check_lines.py
	$(PY) check_scenes.py --strict

serve:
	$(PY) web/server.py

pair:
	$(PY) web/demo_server.py

annotate:
	$(PY) web/annotate_server.py

real:
	$(PY) web/real2_server.py

demo:
	$(PY) run_demo.py mock 3

workbench:
	$(PY) workbench.py data/demo_script.txt --out storyboard.json --ssml --tts --prompt

docker:
	docker build -t manga-director .

run:
	docker run --rm -p 8770:8770 manga-director

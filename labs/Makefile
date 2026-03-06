REPORT_DIR := report
LAB1_DIR := lab1

RPRT := $(REPORT_DIR)/report.pdf

TPST := typst compile --root .
PY3 := python3
ZTHR := zathura

.PHONY: all lab1 report clean

all: lab1 report run-report

lab1:
	@echo "--- Running Lab 1 Source ---"
	@$(PY3) $(LAB1_DIR)/src/main.py

report: $(RPRT)

$(RPRT): $(REPORT_DIR)/report.typ
	@echo "--- Building PDF Report ---"
	@$(TPST) $<

# Просмотр
run-report:
	@echo "--- Opening Report ---"
	@$(ZTHR) $(RPRT) &

clean:
	@echo "--- Cleaning ---"
	$(RM) $(RPRT)
	$(RM) $(LAB1_DIR)/assets/*.png

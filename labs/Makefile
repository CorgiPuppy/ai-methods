REPORT_DIR := report
LAB1_DIR := lab1
LAB2_DIR := lab2

RPRT := $(REPORT_DIR)/report.pdf

TPST := typst compile --font-path ~/../../mnt/c/Windows/Fonts --root .
VENV_DIR := $(PWD)/venv
PY3 := $(VENV_DIR)/bin/python3
ZTHR := zathura

.PHONY: all

all: clean-report run-lab1 run-report

run-lab1:
	@echo "--- Running Lab 1 Source ---"
	@$(PY3) $(LAB1_DIR)/src/main.py

run-lab2:
	@echo "--- Running Lab 2 Source ---"
	. venv/bin/activate
	@$(PY3) $(LAB2_DIR)/src/main.py

run-report: $(RPRT)
	@echo "Running a report.."
	@$(ZTHR) $<	

$(RPRT): $(REPORT_DIR)/*.typ
	@echo "Building pdf.."
	@$(TPST) $<

clean-report: 
	@echo "Cleaning the report.."
	@$(RM) $(RPRT)

REPORT_DIR := report
LAB1_DIR := lab1

RPRT := $(REPORT_DIR)/report.pdf

TPST := typst compile --font-path ~/../../mnt/c/Windows/Fonts --root .
PY3 := python3
ZTHR := zathura

.PHONY: all

all: clean-report run-src run-report

run-src:
	@echo "--- Running Lab 1 Source ---"
	@$(PY3) $(LAB1_DIR)/src/main.py

run-report: $(RPRT)
	@echo "Running a report.."
	@$(ZTHR) $<	

$(RPRT): $(REPORT_DIR)/*.typ
	@echo "Building pdf.."
	@$(TPST) $<

clean-report: 
	@echo "Cleaning the report.."
	@$(RM) $(RPRT)

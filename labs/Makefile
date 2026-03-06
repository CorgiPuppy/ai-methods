REPORT_DIR := report
RPRT := $(REPORT_DIR)/report.pdf

TPST := typst compile --root .
ZTHR := zathura

.PHONY: all

all: clean-report run-report

run-report: $(RPRT)
	@echo "Running a report.."
	@$(ZTHR) $<

$(RPRT): $(REPORT_DIR)/*.typ
	@echo "Building pdf.."
	@$(TPST) $<


clean-report:
	@echo "Cleaning the report.."
	@$(RM) $(RPRT)

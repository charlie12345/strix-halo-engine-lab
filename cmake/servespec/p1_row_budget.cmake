# Row budget across lanes: shared planner (strix_serve) and its host test.
strix_servespec_serve_sources(src/serve/row_budget.cpp)
strix_servespec_host_test(NAME strix-row-budget-tests
    SOURCES tests/row_budget_tests.cpp
    LIBS strix_serve)

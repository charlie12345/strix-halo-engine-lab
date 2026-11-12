# Cycle receipt and ledger (header only) and the serving load harness.
strix_servespec_host_test(NAME strix-cycle-receipt-tests
    SOURCES tests/cycle_receipt_tests.cpp
    LIBS strix_serve_json)

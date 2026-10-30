#pragma once

#include <iostream>
#include <sstream>
#include <string>

// A number for messages: 3 significant digits, scientific when small.
inline std::string sci(double v) {
    std::ostringstream out;
    out.precision(3);
    out << v;
    return out.str();
}

// Named expectations of a test program; main returns failed() != 0.
class Check {
public:
    void expect(bool ok, const std::string& what) {
        std::cout << (ok ? "  ok    " : "  FAIL  ") << what << "\n";
        failed_ += !ok;
    }
    int failed() const { return failed_; }

private:
    int failed_ = 0;
};

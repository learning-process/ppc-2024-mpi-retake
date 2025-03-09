#include <cctype>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>

class SimpleParser {
 public:
  SimpleParser(std::string expr_ession, const std::map<std::string, double>& variables)
      : expr_(std::move(expr_ession)), vars_(variables) {}

  double Parse() { return Parseexpr_ession(); }

 private:
  std::string expr_;
  std::map<std::string, double> vars_;
  size_t pos{0};

  double Parseexpr_ession() {
    double result = ParseTerm();
    while (pos < expr_.length()) {
      if (expr_[pos] == '+') {
        pos++;
        result += ParseTerm();
      } else if (expr_[pos] == '-') {
        pos++;
        result -= ParseTerm();
      } else {
        break;
      }
    }
    return result;
  }

  double ParseTerm() {
    double result = ParseFactor();
    while (pos < expr_.length()) {
      if (expr_[pos] == '*') {
        pos++;
        result *= ParseFactor();
      } else if (expr_[pos] == '/') {
        pos++;
        result /= ParseFactor();
      } else {
        break;
      }
    }
    return result;
  }

  double ParseFactor() {
    if (expr_[pos] == '(') {
      pos++;
      double result = Parseexpr_ession();
      if (expr_[pos] != ')') {
        throw std::runtime_error("Expected ')'");
      }
      pos++;
      return result;
    }
    if (isalpha(expr_[pos]) != 0) {
      return ParseVariable();
    }

    return ParseVariable();
  }

  double ParseVariable() {
    std::string var;
    while (pos < expr_.length() && isalpha(expr_[pos]) != 0) {
      var += expr_[pos];
      pos++;
    }
    if (vars_.find(var) == vars_.end()) {
      throw std::runtime_error("Unknown variable: " + var);
    }
    return vars_[var];
  }

  double ParseNumber() {
    size_t start_pos = pos;
    while (pos < expr_.length() && (isdigit(expr_[pos]) != 0 || expr_[pos] == '.')) {
      pos++;
    }
    return std::stod(expr_.substr(start_pos, pos - start_pos));
  }
};
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

  double Parse() { return ParseExpression(); }

 private:
  std::string expr_;
  std::map<std::string, double> vars_;
  size_t pos_{0};

  double ParseExpression() {
    double result = ParseTerm();
    while (pos_ < expr_.length()) {
      if (expr_[pos_] == '+') {
        pos_++;
        result += ParseTerm();
      } else if (expr_[pos_] == '-') {
        pos_++;
        result -= ParseTerm();
      } else {
        break;
      }
    }
    return result;
  }

  double ParseTerm() {
    double result = ParseFactor();
    while (pos_ < expr_.length()) {
      if (expr_[pos_] == '*') {
        pos_++;
        result *= ParseFactor();
      } else if (expr_[pos_] == '/') {
        pos_++;
        result /= ParseFactor();
      } else {
        break;
      }
    }
    return result;
  }

  double ParseFactor() {
    if (expr_[pos_] == '(') {
      pos_++;
      double result = ParseExpression();
      if (expr_[pos_] != ')') {
        throw std::runtime_error("Expected ')'");
      }
      pos_++;
      return result;
    }
    if (isalpha(expr_[pos_]) != 0) {
      return ParseVariable();
    }

    return ParseVariable();
  }

  double ParseVariable() {
    std::string var;
    while (pos_ < expr_.length() && isalpha(expr_[pos_]) != 0) {
      var += expr_[pos_];
      pos_++;
    }
    if (vars_.find(var) == vars_.end()) {
      throw std::runtime_error("Unknown variable: " + var);
    }
    return vars_[var];
  }

  double ParseNumber() {
    size_t start_pos = pos_;
    while (pos_ < expr_.length() && (isdigit(expr_[pos_]) != 0 || expr_[pos_] == '.')) {
      pos_++;
    }
    return std::stod(expr_.substr(start_pos, pos_ - start_pos));
  }
};
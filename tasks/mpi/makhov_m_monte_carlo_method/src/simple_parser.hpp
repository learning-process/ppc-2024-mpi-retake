#include <cctype>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>

class SimpleParser {
 public:
  SimpleParser(std::string expression, const std::map<std::string, double>& variables)
      : expr(std::move(expression)), vars_(variables), pos(0) {}

  double Parse() { return ParseExpression(); }

 private:
  std::string expr;
  std::map<std::string, double> vars_;
  size_t pos{0};

  double ParseExpression() {
    double result = ParseTerm();
    while (pos < expr.length()) {
      if (expr[pos] == '+') {
        pos++;
        result += ParseTerm();
      } else if (expr[pos] == '-') {
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
    while (pos < expr.length()) {
      if (expr[pos] == '*') {
        pos++;
        result *= ParseFactor();
      } else if (expr[pos] == '/') {
        pos++;
        result /= ParseFactor();
      } else {
        break;
      }
    }
    return result;
  }

  double ParseFactor() {
    if (expr[pos] == '(') {
      pos++;
      double result = ParseExpression();
      if (expr[pos] != ')') {
        throw std::runtime_error("Expected ')'");
      }
      pos++;
      return result;
    }
    if (isalpha(expr[pos])) return parseVariable();

    return parseNumber();
  }

  double parseVariable() {
    std::string var;
    while (pos < expr.length() && isalpha(expr[pos] != 0)) {
      var += expr[pos];
      pos++;
    }
    if (vars_.find(var) == vars_.end()) {
      throw std::runtime_error("Unknown variable: " + var);
    }
    return vars_[var];
  }

  double parseNumber() {
    size_t start_pos = pos;
    while (pos < expr.length() && (isdigit(expr[pos]) != 0 || expr[pos] == '.')) {
      pos++;
    }
    return std::stod(expr.substr(start_pos, pos - start_pos));
  }
};
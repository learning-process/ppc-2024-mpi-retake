#include <string>
#include <map>
#include <cmath>
#include <stdexcept>

class SimpleParser {
public:
    SimpleParser(const std::string& expression, const std::map<std::string, double>& variables)
        : expr(expression), vars(variables), pos(0) {}

    double parse() {
        return parseExpression();
    }

private:
    std::string expr;
    std::map<std::string, double> vars;
    size_t pos;

    double parseExpression() {
        double result = parseTerm();
        while (pos < expr.length()) {
            if (expr[pos] == '+') {
                pos++;
                result += parseTerm();
            } else if (expr[pos] == '-') {
                pos++;
                result -= parseTerm();
            } else {
                break;
            }
        }
        return result;
    }

    double parseTerm() {
        double result = parseFactor();
        while (pos < expr.length()) {
            if (expr[pos] == '*') {
                pos++;
                result *= parseFactor();
            } else if (expr[pos] == '/') {
                pos++;
                result /= parseFactor();
            } else {
                break;
            }
        }
        return result;
    }

    double parseFactor() {
        if (expr[pos] == '(') {
            pos++;
            double result = parseExpression();
            if (expr[pos] != ')') {
                throw std::runtime_error("Expected ')'");
            }
            pos++;
            return result;
        } else if (isalpha(expr[pos])) {
            return parseVariable();
        } else {
            return parseNumber();
        }
    }

    double parseVariable() {
        std::string var;
        while (pos < expr.length() && isalpha(expr[pos])) {
            var += expr[pos];
            pos++;
        }
        if (vars.find(var) == vars.end()) {
            throw std::runtime_error("Unknown variable: " + var);
        }
        return vars[var];
    }

    double parseNumber() {
        size_t startPos = pos;
        while (pos < expr.length() && (isdigit(expr[pos]) || expr[pos] == '.')) {
            pos++;
        }
        return std::stod(expr.substr(startPos, pos - startPos));
    }
};
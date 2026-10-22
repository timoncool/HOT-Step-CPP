// Self-check: visible_when.equals given as a Lua number or boolean is
// stringified the way the UI's String(value) does it. Needs the LuaJIT lib
// from an engine build:
//   cl /std:c++17 /EHsc /utf-8 /I..\src /I..\vendor\luajit\src test-lua-visible-when.cpp ..\build\luajit\src\lua51.lib
#include "lua-plugin.h"
#include <cassert>
#include <cstdio>

static std::string equals_of(const char * lua_value) {
    lua_State * L = luaL_newstate();
    std::string src = std::string("return { key = 'k', visible_when = { key = 'm', equals = ") + lua_value + " } }";
    int rc = luaL_dostring(L, src.c_str());
    assert(rc == 0);
    ParamSchema p = lua_extract_param(L, lua_gettop(L));
    assert(p.visible_when.active && p.visible_when.key == "m");
    lua_close(L);
    return p.visible_when.equals;
}

static void check(const char * lua_value, const char * want) {
    std::string got = equals_of(lua_value);
    if (got != want) {
        std::fprintf(stderr, "equals = %s: got \"%s\", want \"%s\"\n", lua_value, got.c_str(), want);
        assert(false);
    }
}

int main() {
    check("true", "true");
    check("false", "false");
    check("'2.50'", "2.50");  // literal strings pass through untouched
    check("'1e-5'", "1e-5");
    check("2", "2");
    check("2.5", "2.5");
    check("-2.5", "-2.5");
    check("-0.0", "0");
    check("0.1", "0.1");
    check("1/3", "0.3333333333333333");
    check("1e-5", "0.00001");
    check("1e-6", "0.000001");    // last fixed-notation power below 1
    check("1e-7", "1e-7");
    check("1.234e-7", "1.234e-7");
    check("0.000001234", "0.000001234");
    check("1e15", "1000000000000000");
    check("1e20", "100000000000000000000");
    check("1.2345678901234568e20", "123456789012345680000");
    check("1e21", "1e+21");       // first exponent-notation power
    check("1.5e21", "1.5e+21");
    check("-1e21", "-1e+21");
    check("2^-24", "5.960464477539063e-8");  // 16-digit tie; must match Node, not %.16e
    check("5e-324", "5e-324");
    check("1.7976931348623157e308", "1.7976931348623157e+308");
    std::printf("ok\n");
    return 0;
}

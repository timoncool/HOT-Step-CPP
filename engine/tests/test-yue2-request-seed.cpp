// Self-check: yue2_parse_request keeps a uint64 seed exact (JSON integer or
// decimal string). Header-only parse path:
//   cl /std:c++17 /EHsc /I..\src\yue2 /I..\vendor\yyjson test-yue2-request-seed.cpp ..\vendor\yyjson\yyjson.c
#include "yue2-request.h"
#include <cassert>
#include <cstdio>

static Yue2Request parse(const std::string & body, bool * ok, std::string * err) {
    Yue2Request r;
    *ok = yue2_parse_request(body, &r, err);
    return r;
}

int main() {
    bool ok;
    std::string err;
    const uint64_t big = 15213961153666092475ull;
    Yue2Request r = parse(R"({"style":"folk","seed":"15213961153666092475","noise_seed":"15213961153666092476"})", &ok, &err);
    assert(ok && r.seed_present && r.seed == big && r.noise_seed == big + 1);
    r = parse(R"({"style":"folk","seed":15213961153666092475})", &ok, &err);   // integer literal
    assert(ok && r.seed == big);
    r = parse(R"({"style":"folk","seed":12345})", &ok, &err);                  // legacy small number
    assert(ok && r.seed == 12345);
    r = parse(R"({"style":"folk","seed":"18446744073709551616"})", &ok, &err); // past uint64
    assert(!ok);
    r = parse(R"({"style":"folk","seed":18446744073709551616})", &ok, &err);   // numeric overflow
    assert(!ok);
    r = parse(R"({"style":"folk","seed":1e100})", &ok, &err);
    assert(!ok);
    r = parse(R"({"style":"folk","seed":"12x"})", &ok, &err);
    assert(!ok);
    r = parse(R"({"style":"folk","seed":-1})", &ok, &err);
    assert(!ok);
    std::puts("yue2 request seed ok");
    return 0;
}

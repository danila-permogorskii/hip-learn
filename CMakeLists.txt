# =============================================================================
# ck_tile_course — CMakeLists.txt  (HELA REPOT / WHOLE REPO)
# =============================================================================
#
#   Detta ersätter din 8-radiga smoke-only-fil. Skillnaden i ETT ord: skalbarhet.
#   This replaces your 8-line smoke-only file. The difference in ONE word:
#   scalability. Smoke-filen kunde bara bygga ETT mål. Denna fil bygger HELA
#   kursen och gör det trivialt att lägga till nästa rung (one line per rung).
#   The smoke file could build only ONE target. This file builds the WHOLE
#   course and makes adding the next rung trivial.
#
#   MÅL-HÅRDVARA / TARGET HARDWARE
#     Primär (primary):   MI350X, gfx950, CDNA4  — din DigitalOcean VF.
#     Sekundär (secondary): MI300X, gfx942, CDNA3 — FP64-benchmarkmålet.
#   Vi bygger en FET BINÄR (fat binary) som kör på VILKEN SOM HELST av dem.
#   We build a FAT BINARY that runs on WHICHEVER of them is available.
#
#   FILOSOFI / PHILOSOPHY:  "bygg fett, analysera magert"
#                           "build fat, analyse lean"
#   Binären targetar båda arkitekturerna; clangd (.clangd) pinnas till EN.
#   The binary targets both arches; clangd (.clangd) is pinned to ONE.
# =============================================================================

cmake_minimum_required(VERSION 3.21)   # 3.21 = första versionen med HIP-språk
# 3.21 = first version with HIP language
cmake_policy(VERSION 3.21...3.30)


# -----------------------------------------------------------------------------
# KOMPILATOR-PIN / COMPILER PIN  — VIKTIGT, måste stå FÖRE project()
#                                  IMPORTANT, must come BEFORE project()
#
#   PROBLEMET (THE PROBLEM):
#     I ROCm 7.2.x är `hipcc` DEPREKERAD — den vidarebefordrar bara till
#     amdclang++. Om CLion auto-detekterar HIP-kompilatorn kan den välja den
#     gamla hipcc-wrappern, och då blir compile_commands.json "smutsig"
#     (full av wrapper-flaggor som clangd inte förstår).
#     In ROCm 7.2.x, `hipcc` is DEPRECATED — it merely forwards to amdclang++.
#     If CLion auto-detects the HIP compiler it may pick the old hipcc wrapper,
#     making compile_commands.json "dirty" (full of wrapper flags clangd can't
#     parse).
#
#   LÖSNINGEN (THE SOLUTION):
#     Pinna amdclang++ direkt. Då blir varje rad i compile_commands.json ett
#     RENT amdclang++-anrop — exakt vad clangd vill läsa.
#     Pin amdclang++ directly. Then every line in compile_commands.json is a
#     CLEAN amdclang++ invocation — exactly what clangd wants to read.
#
#   NB: detta sätts normalt via CMakePresets.json (cacheVariables), men vi
#       lägger en reserv (fallback) här om någon konfigurerar utan preset.
#       Normally set via CMakePresets.json (cacheVariables), but we add a
#       fallback here in case someone configures without a preset.
# -----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER /opt/rocm/bin/amdclang++ CACHE FILEPATH
            "HIP compiler — amdclang++ (hipcc is deprecated in ROCm 7.2.x)")
endif()


# -----------------------------------------------------------------------------
# PROJEKT-DEKLARATION / PROJECT DECLARATION
#
#   LANGUAGES CXX HIP — vi behöver BÅDA:
#     CXX → ren värdkod (pure host code), t.ex. eventuella hjälpbibliotek.
#     HIP → kärnor och enhetskod (kernels and device code).
#   When CMake sees HIP, it locates hip-lang-config.cmake, sets up the .hip
#   compile rules, and (because we pinned it above) uses amdclang++.
# -----------------------------------------------------------------------------
project(ck_tile_course
        VERSION 0.1.0
        DESCRIPTION "CK Tile learning track — gfx950/gfx942, ROCm 7.2.4"
        LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)


# -----------------------------------------------------------------------------
# FET BINÄR / FAT BINARY  — kärnan i hela uppsättningen
#                           the heart of the whole setup
#
#   En semikolon-separerad lista ger amdclang++ ETT --offload-arch per arch.
#   A semicolon-separated list gives amdclang++ ONE --offload-arch per arch.
#   Resultatet: en enda exekverbar fil som bär kod för BÅDA chipen. Vid körning
#   väljer runtime rätt code object för det chip som faktiskt finns.
#   The result: a single executable carrying code for BOTH chips. At runtime
#   the loader selects the right code object for the chip actually present.
#
#   Detta är VARFÖR du inte behöver byta flagga per session — samma binär kör
#   på MI350X idag och MI300X imorgon utan omkompilering.
#   This is WHY you don't switch a flag per session — the same binary runs on
#   MI350X today and MI300X tomorrow with no recompile.
#
#   Overridbar (overridable): -D CK_ARCHS=gfx942 på kommandoraden eller via
#   en preset för att bygga single-arch (t.ex. för en snabb Debug-build).
#   Override on the command line or via a preset to build single-arch (e.g.
#   for a fast Debug build).
# -----------------------------------------------------------------------------
set(CK_ARCHS "gfx942;gfx950" CACHE STRING
        "Target GPU architectures (semicolon list) — fat binary by default")
set(CMAKE_HIP_ARCHITECTURES "${CK_ARCHS}")


# -----------------------------------------------------------------------------
# FLAGGOR PER BYGGTYP / FLAGS PER BUILD TYPE
#
#   CLion växlar mellan dessa via CMake-profilen (vald preset). Varje profil
#   har ett tydligt syfte — välj rätt verktyg för rätt uppgift.
#   CLion switches between these via the CMake profile (selected preset). Each
#   profile has a clear purpose — the right tool for the right job.
#
#     Debug          → rocgdb. -O0 så varje C++-rad blir en tydlig ISA-sekvens.
#                       -O0 so each C++ line maps to a clear ISA sequence.
#     Release        → benchmarking/rocprofv3. -O3, maximal prestanda.
#     RelWithDebInfo → DAGLIG utveckling. Optimerad MEN stegbar i rocgdb.
#                       DAILY development. Optimised BUT steppable in rocgdb.
#                       (ibland '<optimized out>' — byt till Debug då)
#                       (sometimes '<optimized out>' — switch to Debug then)
#
#   -munsafe-fp-atomics: snabbare atomära flyttal på CDNA; sista biten (ULP)
#     kan variera. Acceptabelt för ML, FARLIGT för FP64-vetenskap — därför
#     INTE i Debug, där du verifierar korrekthet.
#     faster atomic FP on CDNA; the last bit (ULP) may vary. Fine for ML,
#     DANGEROUS for FP64 science — hence NOT in Debug, where you verify
#     correctness.
# -----------------------------------------------------------------------------
set(CMAKE_HIP_FLAGS_DEBUG          "-O0 -g"
        CACHE STRING "HIP flags — Debug")
set(CMAKE_HIP_FLAGS_RELEASE        "-O3 -DNDEBUG -munsafe-fp-atomics"
        CACHE STRING "HIP flags — Release")
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG"
        CACHE STRING "HIP flags — RelWithDebInfo")

# compile_commands.json — clangds livlina (clangd's lifeline). Utan denna ser
# CLion röda streck under __global__, hipMalloc, MFMA-intrinsics. Med den vet
# clangd EXAKT vilka flaggor varje fil byggs med.
# Without it CLion shows red squiggles under __global__/hipMalloc/MFMA. With it
# clangd knows EXACTLY which flags each file is built with.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# -----------------------------------------------------------------------------
# add_experiment() — SKALBARHETS-FUNKTIONEN / THE SCALABILITY FUNCTION
#
#   DETTA är svaret på "en add_executable per fil skalas dåligt".
#   THIS answers "one add_executable per file scales badly".
#
#   Varje övning blir nu EN rad längst ned. Funktionen gör tre saker:
#   Each exercise is now ONE line at the bottom. The function does three things:
#     1. Skapar målet (creates the target).
#     2. Lägger till repo-roten som include-sökväg (gemensamma headers).
#        Adds the repo root as an include path (shared headers).
#     3. Skapar ett valfritt <namn>_asm-mål som dumpar ISA för det målet —
#        ovärderligt när du läser vad MFMA-instruktionen faktiskt blev.
#        Creates an optional <name>_asm target that dumps ISA for that target —
#        invaluable when reading what the MFMA instruction actually became.
#
#   .hip-ändelsen (extension) → ingen LANGUAGE-override behövs; CMake kompilerar
#   .hip automatiskt med HIP-kompilatorn. (De gamla .cpp-experimenten skulle
#   behöva set_source_files_properties(... LANGUAGE HIP) — undvik genom att
#   döpa om till .hip när du migrerar dem.)
#   The .hip extension → no LANGUAGE override needed; CMake compiles .hip with
#   the HIP compiler automatically. (Legacy .cpp experiments would need the
#   property override — avoid it by renaming to .hip when you migrate them.)
# -----------------------------------------------------------------------------
function(add_experiment name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR})

    # ISA-dumpmål (assembly dump target). Bygg med:  cmake --build . --target <name>_asm
    # Genererar <name>.s i byggkatalogen. --save-temps ger även .ll (LLVM IR).
    # Generates <name>.s in the build dir. --save-temps also yields .ll (LLVM IR).
    add_custom_target(${name}_asm
            COMMAND ${CMAKE_HIP_COMPILER}
            $<$<CONFIG:Debug>:-O0> $<$<NOT:$<CONFIG:Debug>>:-O3>
            -std=c++20 -x hip --offload-arch=gfx950 -S
            ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            -o ${CMAKE_BINARY_DIR}/${name}.s
            COMMENT "ISA-dump → ${name}.s (gfx950)"
            VERBATIM)
endfunction()


# -----------------------------------------------------------------------------
# ÖVNINGARNA / THE EXPERIMENTS
#
#   Lägg till en rad per ny rung. Det är HELA underhållsbördan nu.
#   Add one line per new rung. That is the WHOLE maintenance burden now.
# -----------------------------------------------------------------------------
add_experiment(toolchain_smoke   00-toolchain-smoke/toolchain_smoke.hip)
# add_experiment(buffer_view       01-buffer-view/buffer_view.hip)        # Part 2
# add_experiment(tensor_view       02-tensor-view/tensor_view.hip)        # Part 2
# ... climbing toward WarpGemmAttributeMfmaImplF64F64F64M16N16K4 ...

# Bekvämlighetsmål: bygg allt på en gång (build everything at once).
# I CLion: välj "all_experiments" i målväljaren (select it in the target picker).
get_property(_all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
add_custom_target(all_experiments
        COMMENT "Bygger alla experiment / Building all experiments")


# -----------------------------------------------------------------------------
# KONFIGURATIONSSAMMANFATTNING / CONFIGURATION SUMMARY
#   Skrivs ut i CLions CMake-flik vid konfigurering. Snabb sanity-check.
#   Printed in CLion's CMake tab on configure. A quick sanity check.
# -----------------------------------------------------------------------------
message(STATUS "──────────────────────────────────────────────")
message(STATUS " ck_tile_course")
message(STATUS "   Arches (fat):  ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "   HIP compiler:  ${CMAKE_HIP_COMPILER}")
message(STATUS "   Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "   Compile DB:    ${CMAKE_EXPORT_COMPILE_COMMANDS}")
message(STATUS "──────────────────────────────────────────────")
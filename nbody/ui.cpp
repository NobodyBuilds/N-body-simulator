// ui.cpp  —  reworked
//  Changes from original:
//   • syncsettings = true wired up after every widget that touches a settings field.
//     syncSettings() is called once per frame at the end (batched, not per-widget).
//   • Duplicates removed: Speed / Substeps were in Quick + World; now Quick only.
//     Wall force / wall dst were in Quick + Fluid; now Fluid → Forces only.
//   • Each tab is a pure DrawXxxContent() function — no giant monolith.
//   • Hybrid detachable tabs: [^] button floats a tab into a free-floating window;
//     [pin] button re-docks it into the sidebar.  Both modes share the same
//     DrawXxxContent() body so behaviour is identical.
//   • SYNC macro: one line after every widget that modifies settings.

#include <iostream>
#include <cstdarg>
#include "ui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <algorithm>
#include "settings.h"
#include "compute.h"
#include "main.h"
#include<glm/glm.hpp>
// ─────────────────────────────────────────────────────────────────────────────
//  Global sync flag
//  Set to true by any UI widget that modifies a settings field.
//  ui_init() calls syncSettings() at end of frame if true, then clears it.
// ─────────────────────────────────────────────────────────────────────────────
bool syncsettings = false;
bool bchg = false;
// SYNC — drop this after any ImGui widget that writes to settings.
// Uses IsItemEdited() so it fires on every drag/key frame, not just on release.
#define SYNC do { if (ImGui::IsItemEdited()) syncsettings = true; } while(0)

static inline float UIClamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float UIMax(float a, float b) { return a > b ? a : b; }

// ─────────────────────────────────────────────────────────────────────────────
//  Panel geometry
// ─────────────────────────────────────────────────────────────────────────────
static const float PANEL_W = 340.0f;

// ─────────────────────────────────────────────────────────────────────────────
//  Detachable-tab state
//  s_det[i]     — true when tab i is floating as an independent window
//  s_needPos[i] — true once after detach, so we set the initial float position
// ─────────────────────────────────────────────────────────────────────────────
enum {
    TAB_QUICK = 0, TAB_SPH, TAB_NBODY, TAB_PARTICLES,
    TAB_PERF, TAB_HELP,
    TAB_COUNT  // = 6
};
static const char* kTab[TAB_COUNT] = {
    "Quick", "SPH", "N-Body", "Particles", "Perf", "Help"
};
static bool s_det[TAB_COUNT] = {};
static bool s_needPos[TAB_COUNT] = { true,true,true,true,true,true };

// ─────────────────────────────────────────────────────────────────────────────
//  Theme  —  mid-grey, boxy, calm
// ─────────────────────────────────────────────────────────────────────────────
static void ApplyTheme()
{
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 0.0f;  s.ChildRounding = 0.0f;
    s.FrameRounding = 2.0f;  s.PopupRounding = 2.0f;
    s.ScrollbarRounding = 2.0f;  s.GrabRounding = 2.0f;
    s.TabRounding = 2.0f;

    s.WindowPadding = ImVec2(10, 8);   s.FramePadding = ImVec2(6, 3);
    s.ItemSpacing = ImVec2(6, 4);    s.ItemInnerSpacing = ImVec2(4, 3);
    s.IndentSpacing = 14.0f;          s.ScrollbarSize = 9.0f;
    s.GrabMinSize = 8.0f;           s.WindowBorderSize = 1.0f;
    s.FrameBorderSize = 1.0f;           s.TabBorderSize = 0.0f;
    s.SeparatorTextBorderSize = 1.0f;      s.SeparatorTextPadding = ImVec2(6, 2);

    ImVec4* c = s.Colors;
    c[ImGuiCol_Text] = ImVec4(0.87f, 0.87f, 0.85f, 1.00f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.53f, 0.53f, 0.51f, 1.00f);
    c[ImGuiCol_WindowBg] = ImVec4(0.18f, 0.18f, 0.18f, 0.98f);
    c[ImGuiCol_ChildBg] = ImVec4(0.21f, 0.21f, 0.21f, 1.00f);
    c[ImGuiCol_PopupBg] = ImVec4(0.18f, 0.18f, 0.18f, 0.97f);
    c[ImGuiCol_Border] = ImVec4(0.36f, 0.36f, 0.35f, 1.00f);
    c[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_FrameBg] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    c[ImGuiCol_TitleBg] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed] = ImVec4(0.15f, 0.15f, 0.15f, 0.80f);
    c[ImGuiCol_MenuBarBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.40f, 0.40f, 0.39f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.52f, 0.52f, 0.51f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.64f, 0.64f, 0.63f, 1.00f);
    c[ImGuiCol_CheckMark] = ImVec4(0.66f, 0.82f, 0.94f, 1.00f);
    c[ImGuiCol_SliderGrab] = ImVec4(0.54f, 0.70f, 0.84f, 1.00f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.70f, 0.84f, 0.96f, 1.00f);
    c[ImGuiCol_Button] = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
    c[ImGuiCol_Header] = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.48f, 0.48f, 0.48f, 1.00f);
    c[ImGuiCol_Separator] = ImVec4(0.36f, 0.36f, 0.35f, 1.00f);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.54f, 0.54f, 0.53f, 1.00f);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.70f, 0.70f, 0.69f, 1.00f);
    c[ImGuiCol_ResizeGrip] = ImVec4(0.34f, 0.34f, 0.34f, 1.00f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(0.52f, 0.52f, 0.51f, 1.00f);
    c[ImGuiCol_ResizeGripActive] = ImVec4(0.68f, 0.68f, 0.67f, 1.00f);
    c[ImGuiCol_Tab] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    c[ImGuiCol_TabHovered] = ImVec4(0.34f, 0.34f, 0.34f, 1.00f);
    c[ImGuiCol_TabActive] = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
    c[ImGuiCol_TabUnfocused] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    c[ImGuiCol_TabUnfocusedActive] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_PlotLines] = ImVec4(0.58f, 0.68f, 0.76f, 1.00f);
    c[ImGuiCol_PlotLinesHovered] = ImVec4(0.74f, 0.84f, 0.92f, 1.00f);
    c[ImGuiCol_PlotHistogram] = ImVec4(0.50f, 0.62f, 0.72f, 1.00f);
    c[ImGuiCol_PlotHistogramHovered] = ImVec4(0.66f, 0.76f, 0.86f, 1.00f);
    c[ImGuiCol_TableHeaderBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_TableBorderStrong] = ImVec4(0.36f, 0.36f, 0.35f, 1.00f);
    c[ImGuiCol_TableBorderLight] = ImVec4(0.28f, 0.28f, 0.27f, 1.00f);
    c[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.04f);
    c[ImGuiCol_TextSelectedBg] = ImVec4(0.38f, 0.50f, 0.60f, 0.50f);
    c[ImGuiCol_NavHighlight] = ImVec4(0.58f, 0.72f, 0.84f, 1.00f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Widget helpers
// ─────────────────────────────────────────────────────────────────────────────
static void Sec(const char* label)
{
    ImGui::Spacing();
    ImGui::SeparatorText(label);
}

static void Badge(const char* label, ImVec4 col)
{
    ImGui::PushStyleColor(ImGuiCol_Button, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, col);
    ImGui::SmallButton(label);
    ImGui::PopStyleColor(3);
}

static bool DangerButton(const char* label)
{
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.40f, 0.14f, 0.14f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.52f, 0.20f, 0.20f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.64f, 0.26f, 0.26f, 1));
    bool p = ImGui::Button(label, ImVec2(-1, 0));
    ImGui::PopStyleColor(3);
    return p;
}

static void StatRow(const char* label, const char* fmt, ...)
{
    char buf[128];
    va_list a; va_start(a, fmt); vsnprintf(buf, sizeof(buf), fmt, a); va_end(a);
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0); ImGui::TextDisabled("%s", label);
    ImGui::TableSetColumnIndex(1); ImGui::Text("%s", buf);
}

static bool BeginStat2(const char* id)
{
    return ImGui::BeginTable(id, 2,
        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Detach / pin helpers
// ─────────────────────────────────────────────────────────────────────────────

// Right-aligned [^] button that detaches tab id into a floating window.
// Call this as the FIRST item inside BeginChild / BeginTabItem content.
static void DetachButton(int id)
{
    const char* lbl = "[FLOAT]";
    float bw = ImGui::CalcTextSize(lbl).x + ImGui::GetStyle().FramePadding.x * 2.f + 2.f;
    float rx = ImGui::GetContentRegionMax().x - bw;
    ImGui::SetCursorPosX(rx);

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.42f, 0.48f, 0.70f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.52f, 0.56f, 0.62f, 1.00f));

    char bid[16]; snprintf(bid, sizeof(bid), "[^]##D%d", id);
    if (ImGui::SmallButton(bid)) {
        s_det[id] = true;
        s_needPos[id] = true;
    }
    ImGui::PopStyleColor(3);
    ImGui::SetItemTooltip(
        "Float this panel as a movable window.\n"
        "Click [pin] in the floating window to dock it back.");

    ImGui::Separator();
}

// [pin] button + hint text — call at TOP of a floating window's content.
static void PinButton(int id)
{
    if (ImGui::SmallButton("[pin]")) s_det[id] = false;
    ImGui::SetItemTooltip("Dock back into the sidebar tab bar.");
    ImGui::SameLine();
    ImGui::TextDisabled("floating — drag title to move");
    ImGui::Separator();
}

// Set initial position of a newly detached floating window, staggered by id.
static void PrepareFloatWindow(int id)
{
    if (s_needPos[id]) {
        ImGuiIO& io = ImGui::GetIO();
        float ox = ImGui::GetIO().DisplaySize.x * 0.25f + (float)id * 24.f;
        float oy = 70.f + (float)id * 18.f;
        // Keep inside screen bounds
        ox = UIClamp(ox, 10.f, io.DisplaySize.x - 340.f);
        oy = UIClamp(oy, 10.f, io.DisplaySize.y - 520.f);
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_Always);
        s_needPos[id] = false;
    }
    ImGui::SetNextWindowSize(ImVec2(330.f, 500.f), ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.97f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  FPS ring buffer
// ─────────────────────────────────────────────────────────────────────────────
static float s_ring[128] = {};
static int   s_head = 0;

// ─────────────────────────────────────────────────────────────────────────────
//  Particle buffer fill bar (reused in Quick + Particles + Perf)
// ─────────────────────────────────────────────────────────────────────────────


// ═════════════════════════════════════════════════════════════════════════════
//  TAB CONTENT FUNCTIONS
//  Pure content — no Begin/End, no scroll child, no tab wrappers.
//  Called identically whether tab is docked or floating.
// ═════════════════════════════════════════════════════════════════════════════

// ─── QUICK ───────────────────────────────────────────────────────────────────
static void DrawQuickContent()
{
    // ── Simulation control ───────────────────────────────────────────────────
    Sec("Simulation");
    {
        bool paused = !settings.nopause;
        const char* lbl = paused ? "Resume  (Space)" : "Pause  (Space)";
        ImVec4 bc = paused ? ImVec4(0.38f, 0.16f, 0.16f, 1) : ImVec4(0.20f, 0.20f, 0.20f, 1);
        ImVec4 bh = paused ? ImVec4(0.50f, 0.22f, 0.22f, 1) : ImVec4(0.28f, 0.28f, 0.28f, 1);
        ImGui::PushStyleColor(ImGuiCol_Button, bc);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bh);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.44f, 0.44f, 0.44f, 1));
        if (ImGui::Button(lbl, ImVec2(-1, 0))) {
            settings.nopause = !settings.nopause;
            syncsettings = true;
        }
        ImGui::PopStyleColor(3);
        ImGui::SetItemTooltip("Toggle pause / resume.  Same as Space.");
    }

    ImGui::Spacing();
    ImGui::SliderFloat("Speed##q", &settings.simspeed, 0.001f, 10.0f, "%.2f"); SYNC;
    ImGui::SetItemTooltip("dt multiplier.  1.0 = real time.  0.2 = slow-mo.  3.0 = fast-fwd.");
    ImGui::InputInt("Substeps##q", &settings.substeps); SYNC;
    if (settings.substeps < 1) settings.substeps = 1;
    ImGui::SetItemTooltip("Physics steps per frame.  Higher = more stable, linear GPU cost.");

    ImGui::Checkbox("Simulate long run##q", &settings.recordSim); SYNC;
    ImGui::SetItemTooltip("Renders every frame without skipping.  Useful for recording.");

    // ── Mode ─────────────────────────────────────────────────────────────────
    Sec("Mode");
    {
        const char* modes[] = { "N-Body Only", "SPH Only", "N-Body + SPH" };
        if (ImGui::Combo("Mode##q", &settings.mode, modes, IM_ARRAYSIZE(modes))) {
            syncsettings = true;
        }
        ImGui::SetItemTooltip("N-Body: pure gravity.  SPH: fluid sim.  Combined: both forces.");
    }

    // ── N-Body quick params ───────────────────────────────────────────────────
    Sec("N-Body  (see N-Body tab for full detail)");
    ImGui::DragFloat("G##q", &settings.G, 0.01f, 0.0f, 100.0f, "%.3f"); SYNC;
    ImGui::SetItemTooltip("Gravitational constant.  Scales all pairwise attraction.");
    ImGui::DragFloat("Center Mass##q", &settings.centermass, 1.0f, 0.0f, 10000.0f, "%.1f"); SYNC;
    ImGui::SetItemTooltip("Mass of the central attractor body.");
    ImGui::DragFloat("Orbit Speed##q", &settings.orbitspeed, 0.01f, 0.0f, 10.0f, "%.3f"); SYNC;
    ImGui::SetItemTooltip("Initial tangential velocity factor for spawned bodies.");

    // ── SPH quick params (only shown when SPH active) ─────────────────────────
    if (settings.mode != 0)  // not pure N-Body
    {
        Sec("SPH  (see SPH tab for full detail)");
        if (ImGui::SliderFloat("h##q", &settings.h, 0.1f, 20.0f, "%.2f")) {
            calcKernels();
            syncsettings = true;
        }
        ImGui::SetItemTooltip("Smoothing radius.  All kernel coefficients update automatically.");
        ImGui::DragFloat("Rest rho##q", &settings.rest_density, 0.0005f, 0.0f, 1.0f, "%.5f"); SYNC;
        ImGui::SetItemTooltip("Target equilibrium density.");
        ImGui::DragFloat("Stiffness k##q", &settings.pressure, 2.0f, 0.0f, 5000.f, "%.0f"); SYNC;
        ImGui::SetItemTooltip("Compression resistance.");
        ImGui::DragFloat("Viscosity##q", &settings.visc, 0.01f, 0.0f, 200.f, "%.3f"); SYNC;
        ImGui::SetItemTooltip("Low = water.  High = honey.");
    }

    // ── Toggles ──────────────────────────────────────────────────────────────
    Sec("Toggles");
    ImGui::Checkbox("Heat colour##q", &settings.heateffect);  SYNC;
    ImGui::SetItemTooltip("Shift particle colour by kinetic energy.  Blue = slow.  Red = fast.");
    ImGui::SameLine(140);
    ImGui::Checkbox("Debug##q", &settings.debug); SYNC;
    ImGui::SetItemTooltip("Show neighbor count + density stats in HUD.");

    // ── Restart ──────────────────────────────────────────────────────────────
    ImGui::Spacing();
    if (DangerButton("Restart"))
        restartSimulation();
    ImGui::SetItemTooltip("Wipe all particles and respawn with current settings.");

}

// ─── SPH ─────────────────────────────────────────────────────────────────────
static void DrawSPHContent()
{
    // ── Kernel ───────────────────────────────────────────────────────────────
    Sec("Kernel");
    if (ImGui::SliderFloat("h##fl", &settings.h, 0.1f, 20.0f, "%.2f")) {
        calcKernels();
        syncsettings = true;
    }
    ImGui::SetItemTooltip(
        "Interaction radius.  All coefficients derived from this.\n"
        "Larger h = thicker, more cohesive fluid.");

    if (ImGui::TreeNodeEx("Coefficients  (read-only)", ImGuiTreeNodeFlags_SpanFullWidth))
    {
        ImGui::BeginDisabled();
        ImGui::InputFloat("poly6", &settings.pollycoef6, 0, 0, "%.4e");
        ImGui::InputFloat("spiky", &settings.spikycoef, 0, 0, "%.4e");
        ImGui::InputFloat("spiky grad", &settings.spikygradv, 0, 0, "%.4e");
        ImGui::InputFloat("visc lap", &settings.viscosity, 0, 0, "%.4e");
        ImGui::InputFloat("self rho", &settings.Sdensity, 0, 0, "%.4e");
        ImGui::InputFloat("near self", &settings.ndensity, 0, 0, "%.4e");
        ImGui::Text("neighbor count: min:%d  max: %d  avg: %.1f", settings.min_n, settings.max_n, settings.avg_n);
        ImGui::Text("density: min:%.4f  max: %.4f  avg: %.4f", settings.min_density, settings.max_density, settings.avg_density);
        ImGui::Text("near density: min:%.4f  max: %.4f  avg: %.4f", settings.min_neardensity, settings.max_neardensity, settings.avg_neardensity);
        ImGui::EndDisabled();
        ImGui::TreePop();
    }

    // ── Density ──────────────────────────────────────────────────────────────
    Sec("Density");
    ImGui::DragFloat("Rest rho##fl", &settings.rest_density, 0.001f, 0.f, 10000.f, "%.5f"); SYNC;
    ImGui::SetItemTooltip("Target equilibrium density.  Raise to attract particles.  Lower to spread them.");

    // ── Pressure ─────────────────────────────────────────────────────────────
    Sec("Pressure");
    ImGui::DragFloat("Stiffness k", &settings.pressure, 2.f, 0.f, 5000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Compression resistance.  Too high = instability.  Start ~100-500, increase gradually.");
    ImGui::DragFloat("Near k'", &settings.nearpressure, 2.f, 0.f, 10000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Short-range repulsion.  Prevents collapse at close range.  Keep near the same order as k.");


    // ── Viscosity ─────────────────────────────────────────────────────────────
    Sec("Viscosity");
    ImGui::DragFloat("Viscosity##fl", &settings.visc, 0.001f, 0.f, 10.0f, "%.4f"); SYNC;
    ImGui::SetItemTooltip("Velocity averaging between neighbours.  Low = water.  High = honey / thick fluid.");

    // ── Forces ───────────────────────────────────────────────────────────────

    ImGui::DragFloat("cellsize##fl", &settings.cellSize, 0.01f, 0.01f, 4.f, "%.2f"); SYNC;
    ImGui::SetItemTooltip("Spatial hash cell size.  Should match h for best performance.");

    // ── Pipeline toggles ──────────────────────────────────────────────────────
    Sec("Pipeline");
    ImGui::Checkbox("SPH forces##fl", &settings.sph); SYNC;
    ImGui::SetItemTooltip("Master toggle for density + pressure force kernels.\nDisable to watch purely gravity-driven motion.");

    // ── Gravity (SPH context) ─────────────────────────────────────────────────
    Sec("Gravity");
    ImGui::SliderFloat("Gravity##fl", &settings.gravityforce, 0.0f, 1000.0f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Constant downward acceleration per step.  0 = weightless.");

    ImGui::Spacing();
    if (DangerButton("Restart##sph"))
        restartSimulation();
    ImGui::SetItemTooltip("Wipe all particles and respawn.");
}

// ─── N-BODY ──────────────────────────────────────────────────────────────────
static void DrawNBodyContent()
{
    // ── Mode ─────────────────────────────────────────────────────────────────
    Sec("Spawn Mode");
    {
        ImGui::InputInt("spawn mode", &settings.mode); SYNC;
       }

    // ── Gravity ───────────────────────────────────────────────────────────────
    Sec("Gravity");
    ImGui::DragFloat("G##nb", &settings.G, 0.01f, 0.0f, 100.0f, "%.4f"); SYNC;
    ImGui::SetItemTooltip("Gravitational constant.  Scales all pairwise attraction.\nDefault 6.67 (cosmological units).");

    ImGui::DragFloat("Center Mass##nb", &settings.centermass, 1.0f, 0.0f, 100000.0f, "%.1f"); SYNC;
    ImGui::SetItemTooltip("Mass of the fixed central attractor body.\n0 = no central body (free N-body cloud).");

    // ── Orbits / Spawn ────────────────────────────────────────────────────────
    Sec("Orbits  &  Spawn");
    ImGui::DragFloat("Orbit Speed##nb", &settings.orbitspeed, 0.001f, 0.0f, 10.0f, "%.4f"); SYNC;
    ImGui::SetItemTooltip("Tangential velocity multiplier applied at spawn.\n1.0 = circular orbit around center mass.\n<1 = eccentric / infalling.  >1 = escaping.");

    ImGui::DragFloat("Spawn Radius##nb", &settings.radius, 0.5f, 0.0f, 5000.0f, "%.1f"); SYNC;
    ImGui::SetItemTooltip("Inner spawn radius of the particle disk.");

    ImGui::DragFloat("Max Radius##nb", &settings.maxradius, 0.5f, 0.0f, 10000.0f, "%.1f"); SYNC;
    ImGui::SetItemTooltip("Outer edge of the spawn disk.  Particles distributed between Radius and Max Radius.");

    // ── Rendering ─────────────────────────────────────────────────────────────
    Sec("Rendering");
    ImGui::DragFloat("Star Size##nb", &settings.starsize, 0.05f, 0.1f, 100.0f, "%.2f"); SYNC;
    ImGui::SetItemTooltip("Visual point size of the central star body.");

    // ── Debug stats ───────────────────────────────────────────────────────────
    if (settings.debug)
    {
        Sec("Debug  (read-only)");
        ImGui::BeginDisabled();
        ImGui::Text("bodies:  %d active / %d total", settings.count, settings.totalBodies);
        ImGui::Text("substeps: %d   dt: %.5f s", settings.substeps, settings.fixedDt);
        ImGui::EndDisabled();
    }

    ImGui::Spacing();
    if (DangerButton("Restart##nb"))
        restartSimulation();
    ImGui::SetItemTooltip("Respawn all bodies with current settings.");
}

// ─── PARTICLES ───────────────────────────────────────────────────────────────
static void DrawParticlesContent()
{

    // ── Spawn ─────────────────────────────────────────────────────────────────
    Sec("Spawn");
    ImGui::InputInt("Count##pt", &settings.totalBodies);
    if (ImGui::IsItemDeactivatedAfterEdit()) { restartSimulation(); syncsettings = true; }
    ImGui::SetItemTooltip("Particles spawned on restart.  Press Enter to confirm — restarts automatically.");



    ImGui::Spacing();
    ImGui::InputFloat("Radius##pt", &settings.size, 0.05f, 0.2f, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit()) { syncsettings = true; }
    ImGui::SetItemTooltip("Visual + collision radius.  Restart required.");

    ImGui::InputFloat("Mass##pt", &settings.particleMass, 0.1f, 0.5f, "%.3f"); SYNC;

    ImGui::SetItemTooltip("Particle mass.  Affects pressure and density contributions.");

    // ── Spawn tint ────────────────────────────────────────────────────────────


    // ── Heat colour ───────────────────────────────────────────────────────────
    Sec("Heat Colour");
    ImGui::Checkbox("Enable##pth", &settings.heateffect); SYNC;
    ImGui::SetItemTooltip("Shift particle colour by kinetic energy.  Blue = slow.  Red = fast.\n(Invisible in screen-space water render mode — switch to Particles mode.)");
    if (settings.heateffect)
    {
        ImGui::SliderFloat("Gen##pth", &settings.heatMultiplier, 0.1f, 20.f, "%.1f"); SYNC;
        ImGui::SetItemTooltip("How quickly movement raises the heat colour.");
        ImGui::SliderFloat("Fade##pth", &settings.cold, 0.1f, 20.f, "%.1f"); SYNC;
        ImGui::SetItemTooltip("Decay rate — how fast colour cools when idle.");
    }

    ImGui::Spacing();
    if (DangerButton("Restart"))
        restartSimulation();
    ImGui::SetItemTooltip("Wipe all particles and respawn.");

}


// ─── PERF ────────────────────────────────────────────────────────────────────
static void DrawPerfContent()
{
    Sec("Frame Time");

    {
        float  f = settings.avgFps;
        float  n = UIClamp(f / 165.f, 0.f, 1.f);
        ImVec4 col = f > 100 ? ImVec4(0.28f, 0.58f, 0.44f, 1)
            : f > 50 ? ImVec4(0.62f, 0.50f, 0.20f, 1)
            : ImVec4(0.58f, 0.28f, 0.28f, 1);
        char lbl[32]; sprintf(lbl, "%.0f fps", f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
        ImGui::ProgressBar(n, ImVec2(-1, 12), lbl);
        ImGui::PopStyleColor();
        ImGui::SetItemTooltip("Average FPS.  Teal > 100.  Amber > 50.  Red <= 50.");
    }

    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.08f, 0.08f, 1));
    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.52f, 0.64f, 0.74f, 1));
    ImGui::PlotLines("##pfring", s_ring, IM_ARRAYSIZE(s_ring), s_head,
        nullptr, 0.f, 200.f, ImVec2(ImGui::GetContentRegionAvail().x, 44));
    ImGui::PopStyleColor(2);
    ImGui::SetItemTooltip("FPS — last 128 frames.");


    ImGui::Spacing();
    if (BeginStat2("##pffst"))
    {
        StatRow("Avg", "%.1f ms / %.0f fps",
            1000.f / UIMax(settings.avgFps, 0.1f), settings.avgFps);
        StatRow("Min", "%.0f fps", settings.minFps);
        StatRow("Max", "%.0f fps", settings.maxFps);
        ImGui::EndTable();
    }

    Sec("Simulation");
    if (BeginStat2("##pfsst"))
    {
        StatRow("Active", "%d", settings.count);
        StatRow("Substeps", "%d", settings.substeps);
        StatRow("Speed", "%.2fx", settings.simspeed);
        StatRow("dt", "%.5f s", settings.fixedDt);
        ImGui::EndTable();
    }

    ImGui::Spacing();

    Sec("Pipeline  (rough complexity)");
    if (settings.mode == 0 || settings.mode == 2)  // N-Body active
    {
        ImGui::TextDisabled("-- N-Body --");
        ImGui::TextDisabled("computeGravity   O(N^2) or O(N log N)");
        ImGui::TextDisabled("integrateVel     O(N)");
    }
    if (settings.mode == 1 || settings.mode == 2)  // SPH active
    {
        ImGui::TextDisabled("-- SPH --");
        ImGui::TextDisabled("computeHash      O(N)");
        ImGui::TextDisabled("radix sort       O(N log N)");
        ImGui::TextDisabled("reorder          O(N)");
        ImGui::TextDisabled("computeDensity   O(N x K)");
        ImGui::TextDisabled("computePressure  O(N x K)");
    }
    ImGui::TextDisabled("updateKernel     O(N)");
    ImGui::TextDisabled("packVBO          O(N)");
}

// ─── HELP ────────────────────────────────────────────────────────────────────
static void DrawHelpContent()
{
    Sec("Camera");
    if (BeginStat2("##hlcam"))
    {
        StatRow("W / S", "forward / back");
        StatRow("A / D", "strafe left / right");
        StatRow("Q / E", "move up / down");
        StatRow("Mouse", "look around");
        ImGui::EndTable();
    }

    Sec("Controls");
    if (BeginStat2("##hlctrl"))
    {
        StatRow("Space", "pause / resume");
        StatRow("K", "restart simulation");

        StatRow("X", "toggle extra stats (neighbor count, density)");
        ImGui::EndTable();
    }

    Sec("N-Body Parameter Guide");

    auto Tip = [](const char* name, const char* body) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.72f, 0.72f, 0.70f, 1), "%s", name);
        ImGui::Indent(10);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.46f, 0.46f, 0.44f, 1));
        ImGui::TextWrapped("%s", body);
        ImGui::PopStyleColor();
        ImGui::Unindent(10);
        };

    Tip("G — gravitational constant",
        "Scales all pairwise attraction.  Raise to tighten orbits.  Too high = chaotic merging.");
    Tip("Center Mass",
        "Fixed central attractor.  Set 0 to remove it and run a free N-body cloud.");
    Tip("Orbit Speed",
        "Tangential velocity at spawn.  1.0 = circular orbit.  <1 = eccentric/infalling.  >1 = escape trajectory.");
    Tip("Spawn / Max Radius",
        "Inner and outer edge of the initial disk.  Wider gap = more spread-out ring.");
    Tip("Substeps",
        "Gravity is sensitive to large dt.  2-4 substeps strongly recommended.  Linear GPU cost.");

    Sec("SPH Parameter Guide");

    Tip("Performance tip",
        "SPH + N-Body combined is expensive.  Reduce particle count or substeps if framerate drops.");
    Tip("h — smoothing radius",
        "Master kernel scale.  Larger = more neighbours = thicker fluid.  "
        "All coefficients recomputed on change.");
    Tip("Rest density",
        "Equilibrium target.  Raise to compact particles; lower to spread them.");
    Tip("Stiffness k",
        "Compression resistance.  Too high = instability.  "
        "Start low (~100-500), increase gradually while watching substeps.");
    Tip("Near k'",
        "Short-range repulsion.  Raise when particles clump.  In this solver it should usually stay near the same order as k.");
    Tip("Viscosity",
        "Velocity averaging between neighbours.  Low = water.  High = honey.");

    Sec("About");
    ImGui::TextDisabled("N-Body + SPH  —  CUDA + OpenGL");
    ImGui::TextDisabled("Barnes-Hut / direct N^2 gravity");
    ImGui::TextDisabled("Muller poly6 / spiky SPH kernels");
    ImGui::TextDisabled("Symplectic Euler integration");
    ImGui::TextDisabled("CUDA / GL VBO interop");
    ImGui::TextDisabled("Spatial hash  ->  radix sort  ->  reorder");
}

// ═════════════════════════════════════════════════════════════════════════════
//  ui_init  —  called every frame
// ═════════════════════════════════════════════════════════════════════════════
void ui_init()
{
    /*ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();*/
    ApplyTheme();

    // FPS ring update
    s_ring[s_head] = settings.fps;
    s_head = (s_head + 1) % IM_ARRAYSIZE(s_ring);

    ImGuiIO& io = ImGui::GetIO();
    const float SW = io.DisplaySize.x;
    const float SH = io.DisplaySize.y;

    // ── HUD  — top-left corner ────────────────────────────────────────────────
    {
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        const float lh = ImGui::GetTextLineHeight() + 3.0f;
        const float x = 14.0f;
        const float y0 = 10.0f;
        const float pad = 8.0f;

        // ── Pre-format every line so we can measure before drawing ────────────
        struct HudLine { char text[128]; ImU32 col; };
        HudLine lines[8];
        int nLines = 0;

        ImU32 fpsCol = settings.avgFps > 100 ? IM_COL32(96, 188, 148, 210)
            : settings.avgFps > 50 ? IM_COL32(196, 172, 84, 210)
            : IM_COL32(188, 100, 100, 230);
        snprintf(lines[nLines].text, 128, "%.0f fps   min %.0f   max %.0f",
            settings.avgFps, settings.minFps, settings.maxFps);
        lines[nLines++].col = fpsCol;



        snprintf(lines[nLines].text, 128, "cam  %.1f  %.1f  %.1f",
            settings.wx, settings.wy, settings.wz);
        lines[nLines++].col = IM_COL32(120, 180, 140, 155);

        snprintf(lines[nLines].text, 128, "substeps %d   speed %.2fx",
            settings.substeps, settings.simspeed);
        lines[nLines++].col = IM_COL32(120, 120, 118, 148);

        if (settings.debug) {
            snprintf(lines[nLines].text, 128, "neighbor count  min:%d  max:%d  avg:%d",
                settings.min_n, settings.max_n, settings.avg_n);
            lines[nLines++].col = IM_COL32(180, 120, 120, 140);

            snprintf(lines[nLines].text, 128, "density  min:%.4f  max:%.4f  avg:%.4f",
                settings.min_density, settings.max_density, settings.avg_density);
            lines[nLines++].col = IM_COL32(180, 120, 120, 140);

            snprintf(lines[nLines].text, 128, "near density  min:%.4f  max:%.4f  avg:%.4f",
                settings.min_neardensity, settings.max_neardensity, settings.avg_neardensity);
            lines[nLines++].col = IM_COL32(180, 120, 120, 140);
        }

        snprintf(lines[nLines].text, 128, "%s",
            settings.nopause ? "running" : "PAUSED  --  Space to resume");
        lines[nLines++].col = settings.nopause
            ? IM_COL32(96, 168, 128, 170)
            : IM_COL32(188, 100, 100, 230);




        // ── Measure widest line ───────────────────────────────────────────────
        float maxW = 0.0f;
        for (int i = 0; i < nLines; i++) {
            float w = ImGui::CalcTextSize(lines[i].text).x;
            if (w > maxW) maxW = w;
        }

        // ── Dark background rect ──────────────────────────────────────────────
        dl->AddRectFilled(
            ImVec2(x - pad, y0 - pad * 0.5f),
            ImVec2(x + maxW + pad, y0 + nLines * lh + pad * 0.5f),
            IM_COL32(8, 8, 10, 185),
            4.0f   // corner rounding
        );

        // ── Draw text lines ───────────────────────────────────────────────────
        float y = y0;
        for (int i = 0; i < nLines; i++) {
            dl->AddText(ImVec2(x, y), lines[i].col, lines[i].text);
            y += lh;
        }
    }

    if (syncsettings) { syncstruct(); syncsettings = false; }

    // ── Floating detached-tab windows ─────────────────────────────────────────
    //  Rendered BEFORE the sidebar so they sit on top when overlapping.
    typedef void (*DrawFn)();
    static const DrawFn kDraw[TAB_COUNT] = {
        DrawQuickContent, DrawSPHContent, DrawNBodyContent,
        DrawParticlesContent, DrawPerfContent, DrawHelpContent
    };

    for (int i = 0; i < TAB_COUNT; i++)
    {
        if (!s_det[i]) continue;
        PrepareFloatWindow(i);
        if (ImGui::Begin(kTab[i], nullptr,
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav))
        {
            PinButton(i);
            ImGui::BeginChild("##fc", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
            kDraw[i]();
            ImGui::EndChild();
        }
        ImGui::End();
    }

    // ── Main sidebar panel ────────────────────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(SW - PANEL_W, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(PANEL_W, SH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.97f);

    ImGuiWindowFlags wf =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin("##panel", nullptr, wf);

    // Status badge strip
    {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
        const char* modeLabel = (settings.mode == 0) ? "N-BODY"
            : (settings.mode == 1) ? "SPH FLUID"
            : "N-BODY + SPH";
        ImGui::TextDisabled("%s", modeLabel);
        ImGui::SameLine();

        float bx = PANEL_W - 10.0f;

        if (!settings.sph) {
            bx -= 74; ImGui::SetCursorPosX(bx);
            Badge(" SPH OFF ", ImVec4(0.38f, 0.30f, 0.10f, 1));
            ImGui::SameLine();
        }
        if (!settings.nopause) {
            bx -= 62; ImGui::SetCursorPosX(bx);
            Badge(" PAUSED ", ImVec4(0.38f, 0.16f, 0.16f, 1));
        }
    }

    ImGui::Separator();

    // ── Tab bar — detached tabs are hidden ────────────────────────────────────
    bool anyDocked = false;
    for (int i = 0; i < TAB_COUNT; i++) if (!s_det[i]) { anyDocked = true; break; }

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 5));

    if (anyDocked &&
        ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_NoCloseWithMiddleMouseButton))
    {
        ImGui::PopStyleVar();

        // Helper lambda: wraps a tab item + scroll child + detach button + content
        // Using a local macro to avoid repeating the boilerplate 7 times.
        // (C++ lambdas can't easily take a void(*)() param cleanly here,
        //  so we spell out each tab — compiler will inline anyway.)

#define DRAW_TAB(ID, FN) \
        if (!s_det[ID] && ImGui::BeginTabItem(kTab[ID])) { \
            ImGui::BeginChild("##c"#ID, ImVec2(0,0), false, \
                ImGuiWindowFlags_HorizontalScrollbar); \
            DetachButton(ID); \
            FN(); \
            ImGui::EndChild(); \
            ImGui::EndTabItem(); \
        }

        DRAW_TAB(TAB_QUICK, DrawQuickContent)
            DRAW_TAB(TAB_SPH, DrawSPHContent)
            DRAW_TAB(TAB_NBODY, DrawNBodyContent)
            DRAW_TAB(TAB_PARTICLES, DrawParticlesContent)
            DRAW_TAB(TAB_PERF, DrawPerfContent)
            DRAW_TAB(TAB_HELP, DrawHelpContent)

#undef DRAW_TAB

            ImGui::EndTabBar();
    }
    else
    {
        ImGui::PopStyleVar();
        if (!anyDocked) {
            ImGui::Spacing();
            ImGui::TextDisabled("All tabs are floating.");
            ImGui::TextDisabled("Click [pin] in any floating window to restore it here.");
        }
    }


    ImGui::End();  // ##panel


}
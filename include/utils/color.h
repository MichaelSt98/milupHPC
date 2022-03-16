#ifndef CPPUTILS_COLOR_H
#define CPPUTILS_COLOR_H

//! Colors for output formatting
namespace Color {
    enum Code {
        FG_DEFAULT = 39, /// Foreground: distro's default
        FG_BLACK = 30, /// Foreground: black
        FG_RED = 31, /// Foreground: red
        FG_GREEN = 32, /// Foreground: green
        FG_YELLOW = 33, /// Foreground: yellow
        FG_BLUE = 34, /// Foreground: blue
        FG_MAGENTA = 35, /// Foreground: magenta
        FG_CYAN = 36, /// Foreground: cyan
        FG_LIGHT_GRAY = 37, /// Foreground: light gray
        FG_DARK_GRAY = 90, /// Foreground: dark gray
        FG_LIGHT_RED = 91,  /// Foreground: light red
        FG_LIGHT_GREEN = 92, /// Foreground: light green
        FG_LIGHT_YELLOW = 93, /// Foreground: light yellow
        FG_LIGHT_BLUE = 94, /// Foreground: light blue
        FG_LIGHT_MAGENTA = 95, /// Foreground: light magenta
        FG_LIGHT_CYAN = 96, /// Foreground: light cyan
        FG_WHITE = 97, /// Foreground: white
        BG_RED = 41, /// Background: red
        BG_GREEN = 42, /// Background: green
        BG_BLUE = 44, /// Background: blue
        BG_DEFAULT = 49 /// Background: distro's default
    };
}

#endif //CPPUTILS_COLOR_H

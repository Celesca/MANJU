'use strict';

var createTheme = require('../../helpers/create-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var theme = require('../Dropdown/theme.cjs');
var theme$1 = require('../Navbar/theme.cjs');

const megaMenuTheme = createTheme.createTheme({
  ...theme$1.navbarTheme,
  dropdown: {
    base: "",
    toggle: {
      ...theme.dropdownTheme,
      floating: {
        ...theme.dropdownTheme.floating,
        base: tailwindMerge.twMerge(theme.dropdownTheme.floating.base, "mt-2 block"),
        content: tailwindMerge.twMerge(theme.dropdownTheme.floating.content, "text-gray-500 dark:text-gray-400"),
        style: {
          ...theme.dropdownTheme.floating.style,
          auto: tailwindMerge.twMerge(theme.dropdownTheme.floating.style.auto, "text-gray-500 dark:text-gray-400")
        }
      },
      inlineWrapper: tailwindMerge.twMerge(theme.dropdownTheme.inlineWrapper, "flex w-full items-center justify-between")
    }
  },
  dropdownToggle: {
    base: tailwindMerge.twMerge(theme$1.navbarTheme.link.base, theme$1.navbarTheme.link.active.off, "flex w-full items-center justify-between")
  }
});

exports.megaMenuTheme = megaMenuTheme;
//# sourceMappingURL=theme.cjs.map

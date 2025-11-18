'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const PaginationButton = React.forwardRef(
  ({ active, children, className, onClick, theme: customTheme, clearTheme, applyTheme, ...props }, ref) => {
    const provider$1 = provider.useThemeProvider();
    const theme$1 = resolveTheme.useResolveTheme(
      [theme.paginationTheme, provider$1.theme?.pagination, customTheme],
      [get.get(provider$1.clearTheme, "pagination"), clearTheme],
      [get.get(provider$1.applyTheme, "pagination"), applyTheme]
    );
    return /* @__PURE__ */ jsxRuntime.jsx(
      "button",
      {
        ref,
        type: "button",
        className: tailwindMerge.twMerge(active && theme$1.pages.selector.active, className),
        onClick,
        ...props,
        children
      }
    );
  }
);
PaginationButton.displayName = "PaginationButton";
function PaginationNavigation({
  children,
  className,
  onClick,
  disabled = false,
  theme: customTheme,
  clearTheme,
  applyTheme,
  ...props
}) {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.paginationTheme, provider$1.theme?.pagination, customTheme],
    [get.get(provider$1.clearTheme, "pagination"), clearTheme],
    [get.get(provider$1.applyTheme, "pagination"), applyTheme]
  );
  return /* @__PURE__ */ jsxRuntime.jsx(
    "button",
    {
      type: "button",
      className: tailwindMerge.twMerge(disabled && theme$1.pages.selector.disabled, className),
      disabled,
      onClick,
      ...props,
      children
    }
  );
}
PaginationNavigation.displayName = "PaginationNavigation";

exports.PaginationButton = PaginationButton;
exports.PaginationNavigation = PaginationNavigation;
//# sourceMappingURL=PaginationButton.cjs.map

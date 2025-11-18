'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { paginationTheme } from './theme.js';

const PaginationButton = forwardRef(
  ({ active, children, className, onClick, theme: customTheme, clearTheme, applyTheme, ...props }, ref) => {
    const provider = useThemeProvider();
    const theme = useResolveTheme(
      [paginationTheme, provider.theme?.pagination, customTheme],
      [get(provider.clearTheme, "pagination"), clearTheme],
      [get(provider.applyTheme, "pagination"), applyTheme]
    );
    return /* @__PURE__ */ jsx(
      "button",
      {
        ref,
        type: "button",
        className: twMerge(active && theme.pages.selector.active, className),
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
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [paginationTheme, provider.theme?.pagination, customTheme],
    [get(provider.clearTheme, "pagination"), clearTheme],
    [get(provider.applyTheme, "pagination"), applyTheme]
  );
  return /* @__PURE__ */ jsx(
    "button",
    {
      type: "button",
      className: twMerge(disabled && theme.pages.selector.disabled, className),
      disabled,
      onClick,
      ...props,
      children
    }
  );
}
PaginationNavigation.displayName = "PaginationNavigation";

export { PaginationButton, PaginationNavigation };
//# sourceMappingURL=PaginationButton.js.map

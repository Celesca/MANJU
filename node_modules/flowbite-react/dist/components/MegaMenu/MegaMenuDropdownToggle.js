'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef, useId, useRef, useState, useEffect } from 'react';
import { get } from '../../helpers/get.js';
import { mergeRefs } from '../../helpers/merge-refs.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { megaMenuTheme } from './theme.js';

const MegaMenuDropdownToggle = forwardRef((props, ref) => {
  const id = useId();
  const innerRef = useRef(null);
  const [controls, setControls] = useState(void 0);
  const [isExpanded, setExpanded] = useState(void 0);
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [megaMenuTheme.dropdownToggle, provider.theme?.megaMenu?.dropdownToggle, props.theme],
    [get(provider.clearTheme, "megaMenu.dropdownToggle"), props.clearTheme],
    [get(provider.applyTheme, "megaMenu.dropdownToggle"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.megaMenuDropdownToggle);
  function findDropdown() {
    const megaMenu = innerRef.current?.closest("nav");
    return megaMenu?.querySelector('[role="menu"]');
  }
  function onClick() {
    findDropdown()?.classList.toggle("hidden");
    setExpanded(!isExpanded);
  }
  useEffect(() => {
    const dropdown = findDropdown();
    const isDropdownHidden = dropdown?.classList.contains("hidden");
    setControls(dropdown?.id);
    setExpanded(!isDropdownHidden);
  }, []);
  return /* @__PURE__ */ jsx(
    "button",
    {
      ref: mergeRefs([ref, innerRef]),
      "aria-controls": controls,
      "aria-expanded": isExpanded,
      "aria-haspopup": "menu",
      id,
      onClick,
      className: twMerge(theme.base, className),
      ...restProps
    }
  );
});
MegaMenuDropdownToggle.displayName = "MegaMenuDropdownToggle";

export { MegaMenuDropdownToggle };
//# sourceMappingURL=MegaMenuDropdownToggle.js.map

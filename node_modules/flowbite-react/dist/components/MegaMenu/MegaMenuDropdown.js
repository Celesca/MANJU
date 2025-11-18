'use client';
import { jsx } from 'react/jsx-runtime';
import { useState, useId, useRef, useEffect } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Dropdown } from '../Dropdown/Dropdown.js';
import '../Dropdown/DropdownContext.js';
import '../Dropdown/DropdownDivider.js';
import '../Dropdown/DropdownHeader.js';
import '../Dropdown/DropdownItem.js';
import { megaMenuTheme } from './theme.js';

function MegaMenuDropdown(props) {
  const [labelledBy, setLabelledBy] = useState(void 0);
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [megaMenuTheme.dropdown, provider.theme?.megaMenu?.dropdown, props.theme],
    [get(provider.clearTheme, "megaMenu.dropdown"), props.clearTheme],
    [get(provider.applyTheme, "megaMenu.dropdown"), props.applyTheme]
  );
  const { children, className, toggle, ...restProps } = resolveProps(props, provider.props?.megaMenuDropdown);
  if (toggle) {
    return /* @__PURE__ */ jsx(
      Dropdown,
      {
        inline: true,
        label: toggle,
        placement: "bottom",
        theme: theme.toggle,
        className: twMerge(theme.base, className),
        children
      }
    );
  }
  const id = useId();
  const ref = useRef(null);
  useEffect(() => {
    const toggle2 = ref.current?.closest("nav")?.querySelector('[aria-haspopup="menu"]');
    setLabelledBy(toggle2?.id);
  }, []);
  return /* @__PURE__ */ jsx(
    "div",
    {
      "aria-labelledby": labelledBy,
      id,
      ref,
      role: "menu",
      className: twMerge(theme.base, className),
      ...restProps,
      children
    }
  );
}
MegaMenuDropdown.displayName = "MegaMenuDropdown";

export { MegaMenuDropdown };
//# sourceMappingURL=MegaMenuDropdown.js.map

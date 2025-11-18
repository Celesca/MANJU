'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var Dropdown = require('../Dropdown/Dropdown.cjs');
require('../Dropdown/DropdownContext.cjs');
require('../Dropdown/DropdownDivider.cjs');
require('../Dropdown/DropdownHeader.cjs');
require('../Dropdown/DropdownItem.cjs');
var theme = require('./theme.cjs');

function MegaMenuDropdown(props) {
  const [labelledBy, setLabelledBy] = React.useState(void 0);
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.megaMenuTheme.dropdown, provider$1.theme?.megaMenu?.dropdown, props.theme],
    [get.get(provider$1.clearTheme, "megaMenu.dropdown"), props.clearTheme],
    [get.get(provider$1.applyTheme, "megaMenu.dropdown"), props.applyTheme]
  );
  const { children, className, toggle, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.megaMenuDropdown);
  if (toggle) {
    return /* @__PURE__ */ jsxRuntime.jsx(
      Dropdown.Dropdown,
      {
        inline: true,
        label: toggle,
        placement: "bottom",
        theme: theme$1.toggle,
        className: tailwindMerge.twMerge(theme$1.base, className),
        children
      }
    );
  }
  const id = React.useId();
  const ref = React.useRef(null);
  React.useEffect(() => {
    const toggle2 = ref.current?.closest("nav")?.querySelector('[aria-haspopup="menu"]');
    setLabelledBy(toggle2?.id);
  }, []);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "div",
    {
      "aria-labelledby": labelledBy,
      id,
      ref,
      role: "menu",
      className: tailwindMerge.twMerge(theme$1.base, className),
      ...restProps,
      children
    }
  );
}
MegaMenuDropdown.displayName = "MegaMenuDropdown";

exports.MegaMenuDropdown = MegaMenuDropdown;
//# sourceMappingURL=MegaMenuDropdown.cjs.map

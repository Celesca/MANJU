'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var mergeRefs = require('../../helpers/merge-refs.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const MegaMenuDropdownToggle = React.forwardRef((props, ref) => {
  const id = React.useId();
  const innerRef = React.useRef(null);
  const [controls, setControls] = React.useState(void 0);
  const [isExpanded, setExpanded] = React.useState(void 0);
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.megaMenuTheme.dropdownToggle, provider$1.theme?.megaMenu?.dropdownToggle, props.theme],
    [get.get(provider$1.clearTheme, "megaMenu.dropdownToggle"), props.clearTheme],
    [get.get(provider$1.applyTheme, "megaMenu.dropdownToggle"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.megaMenuDropdownToggle);
  function findDropdown() {
    const megaMenu = innerRef.current?.closest("nav");
    return megaMenu?.querySelector('[role="menu"]');
  }
  function onClick() {
    findDropdown()?.classList.toggle("hidden");
    setExpanded(!isExpanded);
  }
  React.useEffect(() => {
    const dropdown = findDropdown();
    const isDropdownHidden = dropdown?.classList.contains("hidden");
    setControls(dropdown?.id);
    setExpanded(!isDropdownHidden);
  }, []);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "button",
    {
      ref: mergeRefs.mergeRefs([ref, innerRef]),
      "aria-controls": controls,
      "aria-expanded": isExpanded,
      "aria-haspopup": "menu",
      id,
      onClick,
      className: tailwindMerge.twMerge(theme$1.base, className),
      ...restProps
    }
  );
});
MegaMenuDropdownToggle.displayName = "MegaMenuDropdownToggle";

exports.MegaMenuDropdownToggle = MegaMenuDropdownToggle;
//# sourceMappingURL=MegaMenuDropdownToggle.cjs.map

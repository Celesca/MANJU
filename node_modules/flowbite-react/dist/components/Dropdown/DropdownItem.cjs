'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var react = require('@floating-ui/react');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var ButtonBase = require('../Button/ButtonBase.cjs');
var DropdownContext = require('./DropdownContext.cjs');
var theme = require('./theme.cjs');

const DropdownItem = React.forwardRef(
  (props, forwardedRef) => {
    const {
      theme: rootTheme,
      clearTheme: rootClearTheme,
      applyTheme: rootApplyTheme,
      activeIndex,
      dismissOnClick,
      getItemProps,
      handleSelect
    } = DropdownContext.useDropdownContext();
    const provider$1 = provider.useThemeProvider();
    const theme$1 = resolveTheme.useResolveTheme(
      [theme.dropdownTheme.floating.item, provider$1.theme?.dropdown?.floating?.item, rootTheme?.floating?.item, props.theme],
      [get.get(provider$1.clearTheme, "dropdown.floating.item"), get.get(rootClearTheme, "floating.item"), props.clearTheme],
      [get.get(provider$1.applyTheme, "dropdown.floating.item"), get.get(rootApplyTheme, "floating.item"), props.applyTheme]
    );
    const {
      children,
      className,
      icon: Icon,
      onClick,
      ...restProps
    } = resolveProps.resolveProps(props, provider$1.props?.dropdownItem);
    const { ref: listItemRef, index } = react.useListItem({ label: typeof children === "string" ? children : void 0 });
    const ref = react.useMergeRefs([forwardedRef, listItemRef]);
    const isActive = activeIndex === index;
    return /* @__PURE__ */ jsxRuntime.jsx("li", { role: "menuitem", className: theme$1.container, children: /* @__PURE__ */ jsxRuntime.jsxs(
      ButtonBase.ButtonBase,
      {
        ref,
        className: tailwindMerge.twMerge(theme$1.base, className),
        ...restProps,
        ...getItemProps({
          onClick: () => {
            onClick?.();
            dismissOnClick && handleSelect(null);
          }
        }),
        tabIndex: isActive ? 0 : -1,
        children: [
          Icon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: theme$1.icon }),
          children
        ]
      }
    ) });
  }
);
DropdownItem.displayName = "DropdownItem";

exports.DropdownItem = DropdownItem;
//# sourceMappingURL=DropdownItem.cjs.map

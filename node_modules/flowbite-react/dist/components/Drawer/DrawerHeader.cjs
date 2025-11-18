'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var closeIcon = require('../../icons/close-icon.cjs');
var homeIcon = require('../../icons/home-icon.cjs');
var provider = require('../../theme/provider.cjs');
var DrawerContext = require('./DrawerContext.cjs');
var theme = require('./theme.cjs');

const DrawerHeader = React.forwardRef((props, ref) => {
  const id = React.useId();
  const {
    id: mainDivId,
    isOpen,
    onClose,
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme
  } = DrawerContext.useDrawerContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.drawerTheme.header, provider$1.theme?.drawer?.header, rootTheme?.header, props.theme],
    [get.get(provider$1.clearTheme, "drawer.header"), get.get(rootClearTheme, "header"), props.clearTheme],
    [get.get(provider$1.applyTheme, "drawer.header"), get.get(rootApplyTheme, "header"), props.applyTheme]
  );
  const {
    children,
    className,
    closeIcon: CloseIcon = closeIcon.CloseIcon,
    title,
    titleIcon: TitleIcon = homeIcon.HomeIcon,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.drawerHeader);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { ref, className, ...restProps, children: [
    /* @__PURE__ */ jsxRuntime.jsxs("h5", { className: theme$1.inner.titleText, id: mainDivId, children: [
      /* @__PURE__ */ jsxRuntime.jsx(TitleIcon, { "aria-hidden": true, className: theme$1.inner.titleIcon }),
      title
    ] }),
    /* @__PURE__ */ jsxRuntime.jsxs("button", { onClick: onClose, type: "button", "data-testid": "close-drawer", className: theme$1.inner.closeButton, children: [
      /* @__PURE__ */ jsxRuntime.jsx(CloseIcon, { "aria-hidden": true, className: theme$1.inner.closeIcon }),
      /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.inner.titleCloseIcon, children: "Close menu" })
    ] }),
    /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.collapsed[isOpen ? "on" : "off"], id: `flowbite-drawer-header-${id}`, children })
  ] });
});
DrawerHeader.displayName = "DrawerHeader";

exports.DrawerHeader = DrawerHeader;
//# sourceMappingURL=DrawerHeader.cjs.map

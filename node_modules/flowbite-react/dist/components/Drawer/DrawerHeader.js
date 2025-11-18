'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef, useId } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { CloseIcon } from '../../icons/close-icon.js';
import { HomeIcon } from '../../icons/home-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useDrawerContext } from './DrawerContext.js';
import { drawerTheme } from './theme.js';

const DrawerHeader = forwardRef((props, ref) => {
  const id = useId();
  const {
    id: mainDivId,
    isOpen,
    onClose,
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme
  } = useDrawerContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [drawerTheme.header, provider.theme?.drawer?.header, rootTheme?.header, props.theme],
    [get(provider.clearTheme, "drawer.header"), get(rootClearTheme, "header"), props.clearTheme],
    [get(provider.applyTheme, "drawer.header"), get(rootApplyTheme, "header"), props.applyTheme]
  );
  const {
    children,
    className,
    closeIcon: CloseIcon$1 = CloseIcon,
    title,
    titleIcon: TitleIcon = HomeIcon,
    ...restProps
  } = resolveProps(props, provider.props?.drawerHeader);
  return /* @__PURE__ */ jsxs("div", { ref, className, ...restProps, children: [
    /* @__PURE__ */ jsxs("h5", { className: theme.inner.titleText, id: mainDivId, children: [
      /* @__PURE__ */ jsx(TitleIcon, { "aria-hidden": true, className: theme.inner.titleIcon }),
      title
    ] }),
    /* @__PURE__ */ jsxs("button", { onClick: onClose, type: "button", "data-testid": "close-drawer", className: theme.inner.closeButton, children: [
      /* @__PURE__ */ jsx(CloseIcon$1, { "aria-hidden": true, className: theme.inner.closeIcon }),
      /* @__PURE__ */ jsx("span", { className: theme.inner.titleCloseIcon, children: "Close menu" })
    ] }),
    /* @__PURE__ */ jsx("span", { className: theme.collapsed[isOpen ? "on" : "off"], id: `flowbite-drawer-header-${id}`, children })
  ] });
});
DrawerHeader.displayName = "DrawerHeader";

export { DrawerHeader };
//# sourceMappingURL=DrawerHeader.js.map

'use client';
import { jsx, jsxs } from 'react/jsx-runtime';
import { useListItem, useMergeRefs } from '@floating-ui/react';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { ButtonBase } from '../Button/ButtonBase.js';
import { useDropdownContext } from './DropdownContext.js';
import { dropdownTheme } from './theme.js';

const DropdownItem = forwardRef(
  (props, forwardedRef) => {
    const {
      theme: rootTheme,
      clearTheme: rootClearTheme,
      applyTheme: rootApplyTheme,
      activeIndex,
      dismissOnClick,
      getItemProps,
      handleSelect
    } = useDropdownContext();
    const provider = useThemeProvider();
    const theme = useResolveTheme(
      [dropdownTheme.floating.item, provider.theme?.dropdown?.floating?.item, rootTheme?.floating?.item, props.theme],
      [get(provider.clearTheme, "dropdown.floating.item"), get(rootClearTheme, "floating.item"), props.clearTheme],
      [get(provider.applyTheme, "dropdown.floating.item"), get(rootApplyTheme, "floating.item"), props.applyTheme]
    );
    const {
      children,
      className,
      icon: Icon,
      onClick,
      ...restProps
    } = resolveProps(props, provider.props?.dropdownItem);
    const { ref: listItemRef, index } = useListItem({ label: typeof children === "string" ? children : void 0 });
    const ref = useMergeRefs([forwardedRef, listItemRef]);
    const isActive = activeIndex === index;
    return /* @__PURE__ */ jsx("li", { role: "menuitem", className: theme.container, children: /* @__PURE__ */ jsxs(
      ButtonBase,
      {
        ref,
        className: twMerge(theme.base, className),
        ...restProps,
        ...getItemProps({
          onClick: () => {
            onClick?.();
            dismissOnClick && handleSelect(null);
          }
        }),
        tabIndex: isActive ? 0 : -1,
        children: [
          Icon && /* @__PURE__ */ jsx(Icon, { className: theme.icon }),
          children
        ]
      }
    ) });
  }
);
DropdownItem.displayName = "DropdownItem";

export { DropdownItem };
//# sourceMappingURL=DropdownItem.js.map

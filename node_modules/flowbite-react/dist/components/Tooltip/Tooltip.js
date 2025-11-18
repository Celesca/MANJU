'use client';
import { jsx } from 'react/jsx-runtime';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Floating } from '../Floating/Floating.js';
import { tooltipTheme } from './theme.js';

function Tooltip(props) {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [tooltipTheme, provider.theme?.tooltip, props.theme],
    [get(provider.clearTheme, "tooltip"), props.clearTheme],
    [get(provider.applyTheme, "tooltip"), props.applyTheme]
  );
  const {
    animation = "duration-300",
    arrow = true,
    children,
    className,
    content,
    placement = "top",
    style = "dark",
    trigger = "hover",
    ...restProps
  } = resolveProps(props, provider.props?.tooltip);
  return /* @__PURE__ */ jsx(
    Floating,
    {
      animation,
      arrow,
      content,
      placement,
      style,
      theme,
      trigger,
      className,
      ...restProps,
      children
    }
  );
}
Tooltip.displayName = "Tooltip";

export { Tooltip };
//# sourceMappingURL=Tooltip.js.map

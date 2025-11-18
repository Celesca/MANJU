'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { avatarTheme } from './theme.js';

const AvatarGroup = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [avatarTheme.group, provider.theme?.avatar?.group, props.theme],
    [get(provider.clearTheme, "avatar"), props.clearTheme],
    [get(provider.applyTheme, "avatar"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.avatarGroup);
  return /* @__PURE__ */ jsx("div", { ref, "data-testid": "avatar-group-element", className: twMerge(theme.base, className), ...restProps });
});
AvatarGroup.displayName = "AvatarGroup";

export { AvatarGroup };
//# sourceMappingURL=AvatarGroup.js.map

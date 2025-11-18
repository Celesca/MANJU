'use client';
import { jsxs } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { avatarTheme } from './theme.js';

const AvatarGroupCounter = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [avatarTheme.groupCounter, provider.theme?.avatar?.groupCounter, props.theme],
    [get(provider.clearTheme, "avatar.groupCounter"), props.clearTheme],
    [get(provider.applyTheme, "avatar.groupCounter"), props.applyTheme]
  );
  const { className, href, total, ...restProps } = resolveProps(props, provider.props?.avatarGroupCounter);
  return /* @__PURE__ */ jsxs("a", { ref, href, className: twMerge(theme.base, className), ...restProps, children: [
    "+",
    total
  ] });
});
AvatarGroupCounter.displayName = "AvatarGroupCounter";

export { AvatarGroupCounter };
//# sourceMappingURL=AvatarGroupCounter.js.map

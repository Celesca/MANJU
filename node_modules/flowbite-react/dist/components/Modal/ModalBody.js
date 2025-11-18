'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useModalContext } from './ModalContext.js';
import { modalTheme } from './theme.js';

const ModalBody = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, popup } = useModalContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [modalTheme.body, provider.theme?.modal?.body, rootTheme?.body, props.theme],
    [get(provider.clearTheme, "modal.body"), get(rootClearTheme, "body"), props.clearTheme],
    [get(provider.applyTheme, "modal.body"), get(rootApplyTheme, "body"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.modalBody);
  return /* @__PURE__ */ jsx("div", { ref, className: twMerge(theme.base, popup && theme.popup, className), ...restProps });
});
ModalBody.displayName = "ModalBody";

export { ModalBody };
//# sourceMappingURL=ModalBody.js.map

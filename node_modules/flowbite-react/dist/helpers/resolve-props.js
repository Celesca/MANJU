import { withoutThemingProps } from './without-theming-props.js';

function resolveProps(props, providerProps) {
  let mergedProps = withoutThemingProps(props);
  if (providerProps) {
    mergedProps = {
      ...providerProps,
      ...props
    };
  }
  return mergedProps;
}

export { resolveProps };
//# sourceMappingURL=resolve-props.js.map

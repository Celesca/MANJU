import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';

const ArrowLeftIcon = forwardRef((props, ref) => /* @__PURE__ */ jsx(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "currentColor",
    stroke: "currentColor",
    strokeWidth: 0,
    viewBox: "0 0 20 20",
    ref,
    ...props,
    children: /* @__PURE__ */ jsx(
      "path",
      {
        fillRule: "evenodd",
        stroke: "none",
        d: "M9.707 16.707a1 1 0 0 1-1.414 0l-6-6a1 1 0 0 1 0-1.414l6-6a1 1 0 0 1 1.414 1.414L5.414 9H17a1 1 0 1 1 0 2H5.414l4.293 4.293a1 1 0 0 1 0 1.414z",
        clipRule: "evenodd"
      }
    )
  }
));
ArrowLeftIcon.displayName = "ArrowLeftIcon";

export { ArrowLeftIcon };
//# sourceMappingURL=arrow-left-icon.js.map

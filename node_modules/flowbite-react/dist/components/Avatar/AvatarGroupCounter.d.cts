import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface AvatarGroupCounterTheme {
    base: string;
}
export interface AvatarGroupCounterProps extends ComponentProps<"a">, ThemingProps<AvatarGroupCounterTheme> {
    total?: number;
}
export declare const AvatarGroupCounter: import("react").ForwardRefExoticComponent<Omit<AvatarGroupCounterProps, "ref"> & import("react").RefAttributes<HTMLAnchorElement>>;

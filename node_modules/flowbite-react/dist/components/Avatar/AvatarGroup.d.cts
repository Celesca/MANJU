import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface AvatarGroupTheme {
    base: string;
}
export interface AvatarGroupProps extends ComponentProps<"div">, ThemingProps<AvatarGroupTheme> {
}
export declare const AvatarGroup: import("react").ForwardRefExoticComponent<Omit<AvatarGroupProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;

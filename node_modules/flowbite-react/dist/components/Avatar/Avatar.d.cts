import { type ComponentProps, type ReactElement } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteBoolean, FlowbiteColors, FlowbitePositions, FlowbiteSizes, ThemingProps } from "../../types";
import type { AvatarGroupTheme } from "./AvatarGroup";
import type { AvatarGroupCounterTheme } from "./AvatarGroupCounter";
export interface AvatarTheme {
    root: AvatarRootTheme;
    group: AvatarGroupTheme;
    groupCounter: AvatarGroupCounterTheme;
}
export interface AvatarRootTheme {
    base: string;
    inner: string;
    bordered: string;
    color: AvatarColors;
    img: AvatarImageTheme;
    initials: AvatarInitialsTheme;
    rounded: string;
    size: AvatarSizes;
    stacked: string;
    status: AvatarStatusTheme;
    statusPosition: FlowbitePositions;
}
export interface AvatarImageTheme extends FlowbiteBoolean {
    base: string;
    placeholder: string;
}
export interface AvatarStatusTheme {
    away: string;
    base: string;
    busy: string;
    offline: string;
    online: string;
}
export interface AvatarInitialsTheme {
    base: string;
    text: string;
}
export interface AvatarColors extends Pick<FlowbiteColors, "failure" | "gray" | "info" | "pink" | "purple" | "success" | "warning"> {
    [key: string]: string;
}
export interface AvatarSizes extends Pick<FlowbiteSizes, "xs" | "sm" | "md" | "lg" | "xl"> {
    [key: string]: string;
}
export interface AvatarImageProps {
    alt?: string;
    className: string;
    "data-testid": string;
}
export interface AvatarProps extends Omit<ComponentProps<"div">, "color">, ThemingProps<AvatarTheme> {
    alt?: string;
    bordered?: boolean;
    img?: string | ((props: AvatarImageProps) => ReactElement);
    color?: DynamicStringEnumKeysOf<AvatarColors>;
    rounded?: boolean;
    size?: DynamicStringEnumKeysOf<AvatarSizes>;
    stacked?: boolean;
    status?: "away" | "busy" | "offline" | "online";
    statusPosition?: keyof FlowbitePositions;
    placeholderInitials?: string;
}
export declare const Avatar: import("react").ForwardRefExoticComponent<Omit<AvatarProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;

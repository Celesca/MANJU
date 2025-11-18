import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface TimelinePointTheme {
    horizontal: string;
    line: string;
    marker: {
        base: {
            horizontal: string;
            vertical: string;
        };
        icon: {
            base: string;
            wrapper: string;
        };
    };
    vertical: string;
}
export interface TimelinePointProps extends ComponentProps<"div">, ThemingProps<TimelinePointTheme> {
    icon?: FC<ComponentProps<"svg">>;
}
export declare const TimelinePoint: import("react").ForwardRefExoticComponent<Omit<TimelinePointProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;

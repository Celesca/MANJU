import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface RatingAdvancedTheme {
    base: string;
    label: string;
    progress: {
        base: string;
        fill: string;
        label: string;
    };
}
export interface RatingAdvancedProps extends ComponentProps<"div">, ThemingProps<RatingAdvancedTheme> {
    percentFilled?: number;
}
export declare const RatingAdvanced: import("react").ForwardRefExoticComponent<Omit<RatingAdvancedProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
